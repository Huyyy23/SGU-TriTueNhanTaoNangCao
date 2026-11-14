"""Evaluate two agents over N games, alternating colours: gate for `az_v0` strength (T21/T31)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.evaluator import evaluate, summarise
from az.game_log import GameLogger
from az.player_factory import wrap_az_player
from config import load_config
from players.az_player import AZPlayer
from players.random import RandomPlayer
from gomoku.rules import get_rules
from players.factory import AGENT_CHOICES
from scripts.tournament import (checkpoint_manifest, checkpoint_mapping, make_named_player,
                                parse_budget, prepare_output, resolve_agent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play agent A against agent B and write summary.json.")
    parser.add_argument("--a", default="random", help="'random' or a checkpoint path.")
    parser.add_argument("--b", default="checkpoints/last.pt", help="'random' or a checkpoint path.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--config", default=None)
    parser.add_argument("--config-a", default=None, help="Config for A; defaults to --config.")
    parser.add_argument("--config-b", default=None, help="Config for B; defaults to --config.")
    parser.add_argument("--out-dir", default=None, help="Defaults to logs/arena/<run_id>/.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rules", choices=("normal", "renju"), default="normal")
    parser.add_argument("--checkpoint", "--checkpoints", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--budget", default=None, help="sims:N or time:S for named agents.")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def make_player(spec: str, rules, cfg: dict, seed: int, time_budget_sec: float | None = None):
    if spec == "random":
        return RandomPlayer(rules, seed=seed)
    player = wrap_az_player(AZPlayer(spec, rules, cfg=cfg, seed=seed, time_budget_sec=time_budget_sec), cfg)
    player.name = f"az:{Path(spec).stem}"
    return player


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.games <= 0 or args.depth <= 0 or args.threads <= 0:
        raise SystemExit("--games, --depth and --threads must be positive")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    out_dir = Path(args.out_dir) if args.out_dir else Path("logs") / "arena" / run_id
    try:
        prepare_output(out_dir)
        budget = parse_budget(args.budget) if args.budget else None
        checkpoints = checkpoint_mapping(args.checkpoint, [args.a, args.b], args.rules)
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    import torch
    torch.set_num_threads(args.threads)

    config_path = args.config or f"configs/{args.rules}.yaml"
    cfg_a = load_config(args.config_a or config_path)
    cfg_b = load_config(args.config_b or config_path)
    rules = get_rules(args.rules)
    players = []
    for spec, config_path, cfg, seed in (
        (args.a, args.config_a or config_path, cfg_a, args.seed),
        (args.b, args.config_b or config_path, cfg_b, args.seed + 10_000),
    ):
        if spec != "random" and resolve_agent(spec) in AGENT_CHOICES:
            if spec == "az_renju" and args.rules != "renju":
                raise SystemExit("az_renju requires --rules renju")
            player = make_named_player(spec, rules, config=config_path, checkpoint=checkpoints.get(spec),
                                       seed=seed, budget=budget or ("sims", int(cfg["n_simulations"])), depth=args.depth)
        elif spec == "random":
            player = make_player(spec, rules, cfg, seed)
        else:
            if budget and budget[0] == "sims":
                cfg["n_simulations"] = int(budget[1])
            player = make_player(spec, rules, cfg, seed,
                                 time_budget_sec=float(budget[1]) if budget and budget[0] == "time" else None)
        players.append(player)
    player_a, player_b = players
    if player_a.name == player_b.name:
        if args.a == "random" or cfg_a == cfg_b:
            raise SystemExit(f"--a and --b are both '{player_a.name}'; the summary cannot tell them apart.")
        player_a.name += ":a"
        player_b.name += ":b"

    manifest_checkpoints = dict(checkpoints)
    for label, spec in (("a", args.a), ("b", args.b)):
        if spec != "random" and resolve_agent(spec) not in AGENT_CHOICES:
            manifest_checkpoints[label] = spec
    manifest = {"config": vars(args), "config_a": cfg_a, "config_b": cfg_b,
                "checkpoints": checkpoint_manifest(manifest_checkpoints)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger = GameLogger(out_dir / "games.jsonl", log_moves=True)
    records = evaluate(player_a, player_b, rules, args.games, logger=logger)
    summary = summarise(records, player_a.name, player_b.name)
    summary["config_a"] = cfg_a
    summary["config_b"] = cfg_b
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
