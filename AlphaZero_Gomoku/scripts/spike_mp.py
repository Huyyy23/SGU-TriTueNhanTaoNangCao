"""Spike (Q6): N processes x M games vs one process x (N*M) games.

Throwaway experiment for GD5: keep only if multiprocessing beats the single-process
batched driver by >= 1.5x wall-clock. Delete this file if it does not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _play(payload: tuple[dict, int, int, str]) -> dict:
    cfg, n_games, seed, checkpoint = payload

    import torch

    torch.set_num_threads(1)

    from az.model import load_checkpoint
    from az.selfplay import run_games
    from gomoku.rules.normal import NormalRules

    model, _ = load_checkpoint(checkpoint)
    started = time.perf_counter()
    records = list(run_games(model, NormalRules(), cfg, n_games, seed=seed))
    wall_sec = time.perf_counter() - started
    return {
        "games": len(records),
        "wall_sec": round(wall_sec, 2),
        "mean_batch": round(sum(r.mean_batch for r in records) / len(records), 3),
        "nn_calls_per_move": round(sum(r.nn_calls for r in records) / sum(r.num_moves for r in records), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Spike: multiprocessing self-play vs one process.")
    parser.add_argument("--config", default="configs/normal.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/last.pt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--games-per-worker", type=int, default=4)
    args = parser.parse_args(argv)

    from az.model import AZNet, load_checkpoint, save_checkpoint
    from az.selfplay import run_games
    from config import load_config
    from gomoku.rules.normal import NormalRules

    cfg = load_config(args.config)

    if not Path(args.checkpoint).is_file():
        args.checkpoint = "logs/spike_model.pt"
        print(f"warning: checkpoint missing, generating {args.checkpoint} from a random-init model")
        save_checkpoint(AZNet(n_blocks=cfg["n_blocks"], n_filters=cfg["n_filters"]), args.checkpoint, meta={"rule_set": "normal"})

    total_games = args.workers * args.games_per_worker

    started = time.perf_counter()
    records = list(run_games(load_checkpoint(args.checkpoint)[0], NormalRules(), {**cfg, "parallel_games": total_games}, total_games, seed=0))
    baseline_sec = time.perf_counter() - started
    baseline = {
        "games": len(records),
        "wall_sec": round(baseline_sec, 2),
        "mean_batch": round(sum(r.mean_batch for r in records) / len(records), 3),
        "nn_calls_per_move": round(sum(r.nn_calls for r in records) / sum(r.num_moves for r in records), 2),
    }

    payloads = [
        ({**cfg, "parallel_games": args.games_per_worker}, args.games_per_worker, seed, str(Path(args.checkpoint).resolve()))
        for seed in range(args.workers)
    ]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        workers = list(pool.map(_play, payloads))
    parallel_sec = time.perf_counter() - started

    games_parallel = sum(w["games"] for w in workers)
    result = {
        "games": total_games,
        "workers": args.workers,
        "games_per_worker": args.games_per_worker,
        "baseline": baseline,
        "multiprocessing": {"wall_sec": round(parallel_sec, 2), "games": games_parallel, "workers": workers},
        "speedup": round(baseline_sec / parallel_sec, 2),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
