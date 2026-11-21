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
from az.logging_v2 import (
    TOURNAMENT_DB,
    connect,
    initialize,
    record_tournament_agent,
    record_tournament_experiment,
    record_tournament_game,
    record_tournament_match,
    tournament_agent_id,
    write_event,
)
from az.player_factory import wrap_az_player
from config import load_config
from players.az_player import AZPlayer
from players.random import RandomPlayer
from gomoku.rules import get_rules
from players.factory import AGENT_CHOICES
from scripts.tournament import (checkpoint_manifest, checkpoint_mapping, make_named_player,
                                parse_budget, resolve_agent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play agent A against agent B and store results in SQLite.")
    parser.add_argument("--a", default="random", help="'random' or a checkpoint path.")
    parser.add_argument("--b", default="checkpoints/normal/latest.json", help="'random' or a checkpoint path.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--config", default=None)
    parser.add_argument("--config-a", default=None, help="Config for A; defaults to --config.")
    parser.add_argument("--config-b", default=None, help="Config for B; defaults to --config.")
    parser.add_argument("--db", type=Path, default=TOURNAMENT_DB, help="Tournament SQLite database.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rules", choices=("normal", "renju"), default="normal")
    parser.add_argument("--checkpoint", "--checkpoints", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--budget", default=None, help="sims:N or time:S for named agents.")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--mcts-backend", choices=("python", "cpp"))
    parser.add_argument("--patterns-backend", choices=("python", "cpp"))
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
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    try:
        budget = parse_budget(args.budget) if args.budget else None
        checkpoints = checkpoint_mapping(args.checkpoint, [args.a, args.b], args.rules)
        from az.model import resolve_checkpoint_path
        checkpoints = {name: str(resolve_checkpoint_path(path)) for name, path in checkpoints.items()}
        for side in ("a", "b"):
            spec = getattr(args, side)
            if spec != "random" and resolve_agent(spec) not in AGENT_CHOICES:
                setattr(args, side, str(resolve_checkpoint_path(spec)))
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
    import torch
    torch.set_num_threads(args.threads)

    config_path = args.config or f"configs/{args.rules}.yaml"
    cfg_a = load_config(args.config_a or config_path)
    cfg_b = load_config(args.config_b or config_path)
    rules = get_rules(args.rules)
    from az.backend import configure_backends
    backends = configure_backends(
        args.backend,
        mcts_backend=args.mcts_backend,
        patterns_backend=args.patterns_backend,
    )
    backend = backends["mcts"]
    patterns_backend_used = backends["patterns"]
    players = []
    for spec, config_path, cfg, seed in (
        (args.a, args.config_a or config_path, cfg_a, args.seed),
        (args.b, args.config_b or config_path, cfg_b, args.seed + 10_000),
    ):
        if spec != "random" and resolve_agent(spec) in AGENT_CHOICES:
            if spec == "az_renju" and args.rules != "renju":
                raise SystemExit("az_renju requires --rules renju")
            player = make_named_player(spec, rules, config=config_path, checkpoint=checkpoints.get(spec),
                                       seed=seed, budget=budget or ("sims", int(cfg["n_simulations"])), depth=args.depth,
                                       backends=backends)
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
    cli_config = vars(args).copy()
    cli_config["db"] = str(args.db)
    cli_config["backend"] = backends
    manifest = {"config": cli_config, "config_a": cfg_a, "config_b": cfg_b,
                "checkpoints": checkpoint_manifest(manifest_checkpoints),
                "backend_metadata": backends}
    db_path = args.db
    initialize(db_path, "tournament")
    agent_configs = {"a": cfg_a, "b": cfg_b}
    agent_specs = {"a": args.a, "b": args.b}
    agent_ids = {}
    with connect(db_path) as connection:
        record_tournament_experiment(
            connection, experiment_id=run_id, started_at=started_at, backend=backend,
            rule_set=args.rules, seed=args.seed, config=manifest,
            metadata={"source": "scripts/arena.py", "backend_metadata": backends},
        )
        for label in ("a", "b"):
            spec = agent_specs[label]
            checkpoint = manifest["checkpoints"].get(label) or manifest["checkpoints"].get(spec)
            algorithm = resolve_agent(spec)
            agent_ids[label] = tournament_agent_id(
                label=label, algorithm=algorithm,
                checkpoint_sha256=checkpoint["sha256"] if checkpoint else None,
                backend=backend, rule_set=args.rules, config=agent_configs[label],
            )
            record_tournament_agent(
                connection, agent_id=agent_ids[label], label=label, algorithm=algorithm,
                checkpoint_sha256=checkpoint["sha256"] if checkpoint else None,
                backend=backend, rule_set=args.rules, config=agent_configs[label],
            )
        match_id = f"{run_id}:a|b"
        record_tournament_match(
            connection, match_id=match_id, experiment_id=run_id,
            agent_a_id=agent_ids["a"], agent_b_id=agent_ids["b"], pair_id="a|b",
            rule_set=args.rules, seed=args.seed, metadata={"games": args.games},
        )

    records = evaluate(player_a, player_b, rules, args.games)
    summary = summarise(records, player_a.name, player_b.name)
    summary["config_a"] = cfg_a
    summary["config_b"] = cfg_b
    with connect(db_path) as connection:
        for record in records:
            payload = {
                **record.__dict__,
                "moves": [move.row * 11 + move.col for move in (record.moves or [])],
                "raw_moves": [move.__dict__ for move in (record.moves or [])],
            }
            record_tournament_game(
                connection, experiment_id=run_id, local_game_id=record.game_id, match_id=match_id,
                black_agent_id=agent_ids["a"] if record.game_id % 2 == 0 else agent_ids["b"],
                white_agent_id=agent_ids["b"] if record.game_id % 2 == 0 else agent_ids["a"],
                record=payload,
            )
        recorded_at = datetime.now(timezone.utc).isoformat()
        write_event(
            connection, experiment_id=run_id, recorded_at=recorded_at,
            event="arena_summary", payload={"summary": summary},
        )
        write_event(
            connection, experiment_id=run_id, recorded_at=recorded_at,
            event="arena_warnings", payload={"warnings": []},
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
