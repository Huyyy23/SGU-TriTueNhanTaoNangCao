"""Measure GD5 batching: batch size and throughput for parallel_games = 1, 4, 16.

Uses the selected validated checkpoint for throughput measurements.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.logging_v2 import SELFPLAY_DB, connect, initialize, record_selfplay_run, write_event

from az.model import load_checkpoint
from az.selfplay import run_games
from config import load_config
from gomoku.rules.normal import NormalRules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark self-play batching.")
    parser.add_argument("--config", default="configs/normal.yaml")
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--parallel", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--checkpoint", default="checkpoints/best.json")
    parser.add_argument("--device", default=None, help="cuda/cpu; default: cuda when available")
    parser.add_argument("--db", default=str(SELFPLAY_DB), help="SQLite self-play database.")
    args = parser.parse_args(argv)
    initialize(args.db, "selfplay")

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from az.model import resolve_checkpoint_path
    args.checkpoint = str(resolve_checkpoint_path(args.checkpoint))
    model, _ = load_checkpoint(args.checkpoint, device=device)
    print(f"device: {device}")
    rules = NormalRules()

    run_id = f"bench_batching_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    created_at = datetime.now(timezone.utc).isoformat()
    rows = []
    for parallel in args.parallel:
        started = time.perf_counter()
        records = list(run_games(model, rules, {**cfg, "parallel_games": parallel}, args.games, seed=0))
        wall_sec = time.perf_counter() - started
        moves = sum(r.moves_count for r in records)
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


    with connect(args.db) as connection:
        record_selfplay_run(connection, run_id=run_id, started_at=created_at, backend="benchmark",
                             rule_set="normal", seed=0,
                             config={"games": args.games, "parallel": args.parallel, **cfg},
                             metadata={"kind": "bench_batching", "device": str(device),
                                       "checkpoint": args.checkpoint})
        for row in rows:
            write_event(connection, run_id=run_id, event="batching_result", payload=row,
                        recorded_at=created_at)
    print(f"wrote {args.db} ({run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
