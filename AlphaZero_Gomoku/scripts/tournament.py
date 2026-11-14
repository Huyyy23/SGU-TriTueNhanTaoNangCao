"""Balanced round-robin evaluation with raw telemetry and reproducible manifests."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.evaluator import play_game, summarise
from az.game_log import GameLogger, GameRecord, MoveRecord
from config import load_config
from gomoku.rules import get_rules
from players.factory import AGENT_CHOICES, create_player

ALIASES = {"mcts_pure": "mcts-pure", "az_v1": "az_v1_threat", "az_renju": "az_v0"}


def resolve_agent(agent: str) -> str:
    return ALIASES.get(agent, agent)


def parse_budget(value: str) -> tuple[str, int | float]:
    try:
        kind, raw = value.split(":", 1)
        amount = int(raw) if kind == "sims" else float(raw)
    except (ValueError, TypeError):
        raise ValueError("budget must be sims:N or time:S with a positive finite value") from None
    if kind not in {"sims", "time"} or not math.isfinite(amount) or amount <= 0:
        raise ValueError("budget must be sims:N or time:S with a positive finite value")
    return kind, amount


def checkpoint_manifest(checkpoints: dict[str, str]) -> dict:
    result = {}
    for name, path in checkpoints.items():
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}
    return result


def prepare_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def checkpoint_mapping(entries: list[str], agents: list[str], rule_set: str) -> dict[str, str]:
    mapping = {}
    for entry in entries:
        for item in entry.split(","):
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                raise ValueError("checkpoint mapping must be NAME=PATH")
            if name in mapping:
                raise ValueError(f"duplicate checkpoint mapping: {name}")
            mapping[name] = path
    if any(not resolve_agent(name).startswith("az") for name in mapping):
        raise ValueError("checkpoint mappings require AZ agents")
    unknown = set(mapping) - set(agents) - {resolve_agent(a) for a in agents}
    if unknown:
        raise ValueError(f"checkpoint mapping names not in agents: {sorted(unknown)}")
    result = {}
    for agent in agents:
        if resolve_agent(agent).startswith("az"):
            default = "checkpoints/renju_last.pt" if rule_set == "renju" or agent == "az_renju" else "checkpoints/last.pt"
            result[agent] = mapping.get(agent, mapping.get(resolve_agent(agent), default))
    return result


def make_named_player(agent: str, rules, *, config: str, checkpoint: str | None,
                      seed: int, budget: tuple[str, int | float], depth: int):
    kind, amount = budget
    player = create_player(
        resolve_agent(agent), rules, config=config, checkpoint=checkpoint, seed=seed,
        rollouts=int(amount) if kind == "sims" else 50, depth=depth,
        n_simulations=int(amount) if kind == "sims" else None,
        time_budget_sec=float(amount) if kind == "time" else None,
    )
    player.name = agent
    return player


def merge_shards(root: Path, out: Path) -> int:
    """Freeze a complete set of pair shards, deriving every result from JSONL."""
    sources = sorted(path.parent for path in root.glob("*/manifest.json"))
    if not sources:
        raise ValueError("merge root contains no pair manifests")
    manifests = [json.loads((path / "manifest.json").read_text(encoding="utf-8")) for path in sources]
    reference = manifests[0]
    config = reference["config"]
    agents = config["agents"]
    expected = list(combinations(agents, 2))
    fields = ("agents", "rules", "budget", "depth", "seed", "games", "loaded_config", "threads")
    checkpoints = {name: item["sha256"] for name, item in reference["checkpoints"].items()}
    games = config["games"]
    if games <= 0 or games % 2:
        raise ValueError("shard games must be positive and even")
    by_pair = {}
    provenance = []
    for source, manifest in zip(sources, manifests):
        cfg = manifest["config"]
        if any(cfg.get(key) != config.get(key) for key in fields):
            raise ValueError(f"incompatible shard configuration: {source}")
        if {name: item["sha256"] for name, item in manifest["checkpoints"].items()} != checkpoints:
            raise ValueError(f"incompatible checkpoint hashes: {source}")
        selected = (cfg.get("pair") or "").split(",")
        pair = next((pair for pair in expected if set(pair) == set(selected)), None)
        if pair is None or pair in by_pair:
            raise ValueError(f"invalid or duplicate pair shard: {source}")
        raw = GameLogger.read_records(source / "games.jsonl")
        if len(raw) != games:
            raise ValueError(f"wrong game count in {source}: {len(raw)}, expected {games}")
        pair_seed = config["seed"] + expected.index(pair) * 100000
        records = []
        for index, item in enumerate(raw):
            a, b = pair
            black, white = (a, b) if index % 2 == 0 else (b, a)
            settings = item.get("config", {})
            seed = pair_seed + index * 2
            if (item["game_id"] != index or item["first_player"] != black or item["second_player"] != white
                    or item["rule_set"] != config["rules"] or item["winner"] not in {"black", "white", "draw"}
                    or settings.get("pair") != list(pair) or settings.get("pair_seed") != pair_seed
                    or settings.get("seed_a") != seed or settings.get("seed_b") != seed + 1
                    or settings.get("budget") != config["budget"] or settings.get("depth") != config["depth"]):
                raise ValueError(f"invalid game identity, colors or seeds: {source}, game {index}")
            if item.get("moves") is None:
                raise ValueError(f"missing raw moves: {source}, game {index}")
            item["moves"] = [MoveRecord(**move) for move in item["moves"]]
            record = GameRecord(**item)
            record.validate()
            records.append(record)
        by_pair[pair] = records
        provenance.append({"path": str(source.resolve()),
                           "games_sha256": hashlib.sha256((source / "games.jsonl").read_bytes()).hexdigest(),
                           "manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()})
    if set(by_pair) != set(expected):
        raise ValueError(f"missing pair shards: {sorted(set(expected) - set(by_pair))}")
    prepare_output(out)
    merged_manifest = {**reference, "created_at": datetime.now(timezone.utc).isoformat(),
                       "config": {**config, "pair": None, "out_dir": str(out)}, "sources": provenance}
    (out / "manifest.json").write_text(json.dumps(merged_manifest, indent=2), encoding="utf-8")
    logger = GameLogger(out / "games.jsonl", log_moves=True)
    matrix = {a: {} for a in agents}
    pairs = []
    count = 0
    for pair in expected:
        records = by_pair[pair]
        for record in records:
            record.game_id = count
            logger.log_game(record)
            count += 1
        a, b = pair
        result = summarise(records, a, b)
        result["seed"] = config["seed"] + expected.index(pair) * 100000
        pairs.append(result)
        matrix[a][b], matrix[b][a] = result["score"], 1 - result["score"]
    summary = {"agents": agents, "total_games": count, "pairs": pairs, "matrix": matrix,
               "manifest": "manifest.json", "latency_percentile": "nearest-rank p95",
               "score_interval": "95% Wilson; draws count as half a success"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["agent", *agents])
        for agent in agents:
            writer.writerow([agent, *(matrix[agent].get(other, "") for other in agents)])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", default=None, help="Comma-separated factory names or aliases.")
    parser.add_argument("--games", type=int, default=100, help="Even number of games per pair.")
    parser.add_argument("--budget", default="sims:200")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--rules", "--rule-set", choices=("normal", "renju"), default="normal")
    parser.add_argument("--checkpoint", "--checkpoints", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--pair", default=None, help="Run only NAME,NAME, preserving full-lineup pair seeds.")
    parser.add_argument("--merge-root", default=None, help="Merge complete pair shard directories without replay.")
    parser.add_argument("--resume", action="store_true", help="Continue one existing pair without replaying logged games.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.merge_root:
        if not args.out_dir:
            parser.error("--merge-root requires --out-dir")
        try:
            return merge_shards(Path(args.merge_root), Path(args.out_dir))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            parser.error(str(exc))
    if not args.agents:
        parser.error("--agents is required unless merging shards")
    agents = [a.strip() for a in args.agents.split(",")]
    try:
        if len(agents) < 2 or len(set(agents)) != len(agents) or any(not a for a in agents):
            raise ValueError("at least two distinct nonempty agent names required")
        if any(resolve_agent(a) not in AGENT_CHOICES for a in agents):
            raise ValueError(f"unknown agent; choose from {AGENT_CHOICES} or {tuple(ALIASES)}")
        if "az_renju" in agents and args.rules != "renju":
            raise ValueError("az_renju requires --rules renju")
        if args.games <= 0 or args.games % 2:
            raise ValueError("--games must be positive and even for balanced colors")
        if args.depth <= 0 or args.threads <= 0:
            raise ValueError("--depth and --threads must be positive")
        selected_pair = args.pair.split(",") if args.pair else None
        if selected_pair and (len(selected_pair) != 2 or len(set(selected_pair)) != 2 or not set(selected_pair) <= set(agents)):
            raise ValueError("--pair must contain two distinct names from --agents")
        budget = parse_budget(args.budget)
        config_path = args.config or f"configs/{args.rules}.yaml"
        config = load_config(config_path)
        checkpoints = checkpoint_mapping(args.checkpoint, agents, args.rules)
        hashes = checkpoint_manifest(checkpoints)
        rules = get_rules(args.rules)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        out = Path(args.out_dir) if args.out_dir else Path("logs/tournament") / run_id
        if not args.resume:
            prepare_output(out)
        elif not args.out_dir or not selected_pair:
            raise ValueError("--resume requires --out-dir and --pair")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    import torch
    torch.set_num_threads(args.threads)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {**vars(args), "agents": agents, "config_path": config_path, "loaded_config": config},
        "checkpoints": hashes,
        "telemetry_semantics": {"nn_calls": "evaluated neural leaves, not batched forward-pass calls",
                                "latency_sec": "select_move wall time; excludes player construction and legality validation"},
        "seed_scheme": "pair_seed = seed + pair_index * 100000; game_seed = pair_seed + game_index * 2; opponent_seed = game_seed + 1",
    }
    existing = []
    if args.resume:
        saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        fields = ("agents", "rules", "budget", "depth", "seed", "games", "loaded_config", "threads", "pair")
        if any(saved["config"].get(key) != manifest["config"].get(key) for key in fields):
            parser.error("resume configuration does not match saved manifest")
        if saved["checkpoints"] != hashes:
            parser.error("resume checkpoint does not match saved manifest")
        existing = GameLogger.read_records(out / "games.jsonl") if (out / "games.jsonl").exists() else []
        if len(existing) > args.games:
            parser.error("resume log contains too many games")
        pair_index, (a, b) = next((i, pair) for i, pair in enumerate(combinations(agents, 2)) if set(pair) == set(selected_pair))
        pair_seed = args.seed + pair_index * 100000
        for index, item in enumerate(existing):
            black, white = (a, b) if index % 2 == 0 else (b, a)
            settings = item.get("config", {})
            seed = pair_seed + index * 2
            if (item["game_id"] != index or item["first_player"] != black or item["second_player"] != white
                    or item["rule_set"] != args.rules or item["winner"] not in {"black", "white", "draw"}
                    or settings.get("pair") != [a, b] or settings.get("pair_seed") != pair_seed
                    or settings.get("seed_a") != seed or settings.get("seed_b") != seed + 1
                    or settings.get("budget") != args.budget or settings.get("depth") != args.depth
                    or item.get("moves") is None):
                parser.error(f"invalid resume game identity or seed at index {index}")
            item["moves"] = [MoveRecord(**move) for move in item["moves"]]
            GameRecord(**item).validate()
    else:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger = GameLogger(out / "games.jsonl", log_moves=True)
    pairs = []
    matrix = {a: {} for a in agents}
    game_id = len(existing)
    for pair_index, (a, b) in enumerate(combinations(agents, 2)):
        if selected_pair and {a, b} != set(selected_pair):
            continue
        pair_seed = args.seed + pair_index * 100000
        records = [GameRecord(**item) for item in existing]
        for index in range(len(existing), args.games):
            seed = pair_seed + index * 2
            pa = make_named_player(a, rules, config=config_path, checkpoint=checkpoints.get(a), seed=seed, budget=budget, depth=args.depth)
            pb = make_named_player(b, rules, config=config_path, checkpoint=checkpoints.get(b), seed=seed + 1, budget=budget, depth=args.depth)
            black, white = (pa, pb) if index % 2 == 0 else (pb, pa)
            record = play_game(black, white, rules, game_id)
            record.config = {"pair": [a, b], "pair_seed": pair_seed, "seed_a": seed, "seed_b": seed + 1,
                             "budget": args.budget, "depth": args.depth, "config_path": config_path}
            logger.log_game(record)
            records.append(record)
            game_id += 1
            print(f"{a} vs {b}: {index + 1}/{args.games} winner={record.winner}", flush=True)
        pair = summarise(records, a, b)
        pair["seed"] = pair_seed
        pairs.append(pair)
        matrix[a][b] = pair["score"]
        matrix[b][a] = 1 - pair["score"]
        summary = {"agents": agents, "total_games": game_id, "pairs": pairs, "matrix": matrix,
                   "manifest": "manifest.json", "latency_percentile": "nearest-rank p95",
                   "score_interval": "95% Wilson; draws count as half a success"}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        with (out / "matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["agent", *agents])
            for agent in agents:
                writer.writerow([agent, *(matrix[agent].get(other, "") for other in agents)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
