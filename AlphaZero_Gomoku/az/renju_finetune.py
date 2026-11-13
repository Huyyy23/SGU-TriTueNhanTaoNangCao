from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from az.replay_buffer import ReplayBuffer
from az.selfplay import run_games
from az.trainer import Trainer
from config import load_config
from gomoku.rules import get_rules


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune AlphaZero checkpoint on Renju self-play.")
    parser.add_argument("--config", default="configs/renju.yaml")
    parser.add_argument("--source", default="checkpoints/last.pt")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_file():
        raise SystemExit(f"source checkpoint not found: {source}")
    cfg = load_config(args.config)
    if cfg["rule_set"] != "renju":
        raise SystemExit("fine-tuning config must set rule_set: renju")
    if args.iterations is not None:
        cfg["n_iter"] = args.iterations
    if args.games is not None:
        cfg["games_per_iter"] = args.games

    out_dir, log_dir = Path(args.out_dir), Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "renju_last.pt"
    buffer_path = out_dir / "renju_buffer_last.npz"
    log_path = log_dir / f"renju_finetune_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    rules = get_rules("renju")
    buffer = ReplayBuffer(cfg["buffer_size"])
    trainer = Trainer.from_pretrained(source, cfg, buffer=buffer, seed=args.seed)

    for iteration in range(cfg["n_iter"]):
        records = list(run_games(
            trainer.model, rules, cfg, cfg["games_per_iter"], seed=args.seed + iteration,
            buffer=buffer, model_iter=iteration,
        ))
        if trainer.eval_batch is None:
            trainer.freeze_eval_batch()
        stats = trainer.train_steps()
        eval_loss = trainer.loss_on_fixed_batch()
        trainer.n_iter = iteration + 1
        trainer.save(checkpoint, buffer_path)
        row = {
            "iteration": trainer.n_iter,
            "rule_set": "renju",
            "source_checkpoint": str(source.resolve()),
            "games": len(records),
            "buffer_size": len(buffer),
            "loss": stats["loss"],
            "policy_loss": stats["policy_loss"],
            "value_loss": stats["value_loss"],
            "loss_eval_on_fixed_batch": eval_loss,
            "nn_calls_per_game": sum(r.nn_calls for r in records) / len(records),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
        print(json.dumps(row))
    print(f"done: {trainer.n_iter} iterations; checkpoint={checkpoint}; log={log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
