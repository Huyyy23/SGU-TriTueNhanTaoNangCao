"""Measure GD5 batching: batch size and throughput for parallel_games = 1, 4, 16.

Uses the trained checkpoint when it exists (throughput is the same either way).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.model import AZNet, load_checkpoint
from az.selfplay import run_games
from config import load_config
from gomoku.rules.normal import NormalRules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark self-play batching.")
    parser.add_argument("--config", default="configs/normal.yaml")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--parallel", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--checkpoint", default="checkpoints/last.pt")
    parser.add_argument("--device", default=None, help="cuda/cpu; default: cuda when available")
    parser.add_argument("--out", default="logs/bench_batching.json")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Path(args.checkpoint).is_file():
        model, _ = load_checkpoint(args.checkpoint, device=device)
    else:
        print(f"warning: {args.checkpoint} missing, using a random-init model")
        model = AZNet(n_blocks=cfg["n_blocks"], n_filters=cfg["n_filters"]).to(device)
    print(f"device: {device}")
    rules = NormalRules()

    rows = []
    for parallel in args.parallel:
        started = time.perf_counter()
        records = list(run_games(model, rules, {**cfg, "parallel_games": parallel}, args.games, seed=0))
        wall_sec = time.perf_counter() - started
        moves = sum(r.num_moves for r in records)
        row = {
            "parallel_games": parallel,
            "games": len(records),
            "wall_sec": round(wall_sec, 2),
            "games_per_hour": round(len(records) / wall_sec * 3600, 1),
            "mean_batch": round(sum(r.mean_batch for r in records) / len(records), 3),
            "nn_calls_per_move": round(sum(r.nn_calls for r in records) / moves, 2),
            "mean_moves_per_game": round(moves / len(records), 2),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
