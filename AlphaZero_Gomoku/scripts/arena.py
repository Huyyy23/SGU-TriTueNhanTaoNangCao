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
from gomoku.rules.normal import NormalRules
from players.az_player import AZPlayer
from players.random import RandomPlayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play agent A against agent B and write summary.json.")
    parser.add_argument("--a", default="random", help="'random' or a checkpoint path.")
    parser.add_argument("--b", default="checkpoints/last.pt", help="'random' or a checkpoint path.")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--config", default="configs/normal.yaml")
    parser.add_argument("--config-a", default=None, help="Config for A; defaults to --config.")
    parser.add_argument("--config-b", default=None, help="Config for B; defaults to --config.")
    parser.add_argument("--out-dir", default=None, help="Defaults to logs/arena/<run_id>/.")
    parser.add_argument("--seed", type=int, default=0)
    return parser


def make_player(spec: str, rules, cfg: dict, seed: int):
    if spec == "random":
        return RandomPlayer(rules, seed=seed)
    player = wrap_az_player(AZPlayer(spec, rules, cfg=cfg, seed=seed), cfg)
    player.name = f"az:{Path(spec).stem}"
    return player


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("logs") / "arena" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_a = load_config(args.config_a or args.config)
    cfg_b = load_config(args.config_b or args.config)
    rules = NormalRules()
    player_a = make_player(args.a, rules, cfg_a, seed=args.seed)
    player_b = make_player(args.b, rules, cfg_b, seed=args.seed + 10_000)
    if player_a.name == player_b.name:
        if args.a == "random" or cfg_a == cfg_b:
            raise SystemExit(f"--a and --b are both '{player_a.name}'; the summary cannot tell them apart.")
        player_a.name += ":a"
        player_b.name += ":b"

    logger = GameLogger(out_dir / "games.jsonl")
    records = evaluate(player_a, player_b, rules, args.games, logger=logger)
    summary = summarise(records, player_a.name, player_b.name)
    summary["config_a"] = cfg_a
    summary["config_b"] = cfg_b
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
