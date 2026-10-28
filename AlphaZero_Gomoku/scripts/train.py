"""Minimal GD4 training driver: self-play -> buffer -> train -> checkpoint.

B owns this file for GD4 (T12). C takes it over at T14, which adds device/worker/time-budget flags.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.game_log import GameLogger
from az.model import AZNet
from az.replay_buffer import ReplayBuffer
from az.selfplay import run_games
from az.trainer import Trainer
from config import load_config
from gomoku.rules.normal import NormalRules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AlphaZero Gomoku agent.")
    parser.add_argument("--config", default="configs/normal.yaml")
    parser.add_argument("--iterations", type=int, default=None, help="Total iteration budget (overrides cfg n_iter).")
    parser.add_argument("--resume", action="store_true", help="Continue from last.pt + buffer_last.npz.")
    parser.add_argument("--out-dir", default="checkpoints", help="Where last.pt and buffer_last.npz go.")
    parser.add_argument("--log-dir", default="logs", help="Where train_<run_id>.jsonl goes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--games", type=int, default=None, help="Games per iteration (overrides cfg games_per_iter).")
    parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations per move (overrides cfg).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    if args.iterations is not None:
        cfg["n_iter"] = args.iterations
    if args.games is not None:
        cfg["games_per_iter"] = args.games
    if args.simulations is not None:
        cfg["n_simulations"] = args.simulations
    if cfg["rule_set"] != "normal":
        raise SystemExit(f"rule_set '{cfg['rule_set']}' has no rules implementation yet (T24).")

    rules = NormalRules()
    out_dir = Path(args.out_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "last.pt"
    buffer_path = out_dir / "buffer_last.npz"
    log_path = log_dir / f"train_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    games_logger = GameLogger(log_path.with_name(log_path.name.replace("train_", "games_")), log_moves=cfg["log_moves"])

    if args.resume:
        if not checkpoint_path.is_file():
            raise SystemExit(f"--resume needs a checkpoint at {checkpoint_path}")
        if buffer_path.is_file():
            buffer = ReplayBuffer.load_npz(buffer_path)
        else:
            print(f"warning: {buffer_path} missing, resuming with an empty buffer")
            buffer = ReplayBuffer(cfg["buffer_size"])
        trainer = Trainer.load(checkpoint_path, cfg, buffer=buffer, seed=args.seed)
        print(f"resumed at iteration {trainer.n_iter}, global_step {trainer.global_step}")
    else:
        buffer = ReplayBuffer(cfg["buffer_size"])
        trainer = Trainer(AZNet(n_blocks=cfg["n_blocks"], n_filters=cfg["n_filters"]), cfg, buffer=buffer, seed=args.seed)

    for iteration in range(trainer.n_iter, cfg["n_iter"]):
        selfplay_started = time.perf_counter()
        records = list(run_games(trainer.model, rules, cfg, cfg["games_per_iter"], seed=args.seed + iteration, buffer=buffer))
        selfplay_sec = time.perf_counter() - selfplay_started

        # One JSONL line per game, so every game of the run leaves a trace (T19 schema).
        for record in records:
            games_logger.log_game(replace(record, game_id=iteration * int(cfg["games_per_iter"]) + record.game_id))

        if trainer.eval_batch is None and len(buffer) > 0:
            trainer.freeze_eval_batch()

        train_started = time.perf_counter()
        stats = trainer.train_steps()
        train_sec = time.perf_counter() - train_started
        eval_loss = trainer.loss_on_fixed_batch()

        trainer.n_iter = iteration + 1
        trainer.save(checkpoint_path, buffer_path)

        games = len(records)
        row = {
            "iteration": trainer.n_iter,
            "selfplay_sec": round(selfplay_sec, 2),
            "train_sec": round(train_sec, 2),
            "games": games,
            "buffer_size": len(buffer),
            "distinct_samples": stats["distinct_samples"],
            "loss": round(stats["loss"], 5),
            "policy_loss": round(stats["policy_loss"], 5),
            "value_loss": round(stats["value_loss"], 5),
            "loss_eval_on_fixed_batch": round(eval_loss, 5),
            "mean_moves_per_game": round(sum(r.num_moves for r in records) / games, 2),
            "nn_calls_per_game": round(sum(r.nn_calls for r in records) / games, 1),
            "mean_batch": round(sum(r.mean_batch for r in records) / games, 3),
            "global_step": trainer.global_step,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row))

    print(f"done: {trainer.n_iter} iterations, checkpoint {checkpoint_path}, log {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
