from __future__ import annotations

import argparse
import json
import uuid
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
    parser.add_argument("--source", default="checkpoints/best.json")
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument("--db", default="data/selfplay.sqlite")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--games", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--mcts-backend", choices=("python", "cpp"))
    parser.add_argument("--patterns-backend", choices=("python", "cpp"))
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_file():
        raise SystemExit(f"source checkpoint not found: {source}")
    from az.model import resolve_checkpoint_path
    source = resolve_checkpoint_path(source)
    cfg = load_config(args.config)
    if cfg["rule_set"] != "renju":
        raise SystemExit("fine-tuning config must set rule_set: renju")
    if args.iterations is not None:
        cfg["n_iter"] = args.iterations
    if args.games is not None:
        cfg["games_per_iter"] = args.games
    from az.backend import configure_backends
    patterns_backend = args.patterns_backend
    if args.backend == "auto" and patterns_backend is None:
        patterns_backend = "python"
    try:
        backends = configure_backends(args.backend, mcts_backend=args.mcts_backend,
                                      patterns_backend=patterns_backend)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"backends": backends}, sort_keys=True))
    cfg["backend"] = args.backend
    cfg["backend_metadata"] = backends
    cfg["mcts_backend"] = backends["mcts"]
    cfg["patterns_backend"] = backends["patterns"]
    cfg["components_used"] = {"mcts": backends["mcts"], "patterns": "not_used"}

    out_dir = Path(args.out_dir)
    db_path = Path(args.db)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "renju_latest.json"
    run_id = f"renju_finetune_{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}"
    started_at = datetime.now(timezone.utc).isoformat()
    from az import logging_v2
    logging_v2.initialize(db_path, "selfplay")
    db = logging_v2.connect(db_path)
    with db:
        logging_v2.record_selfplay_run(
            db, run_id=run_id, started_at=started_at, backend=args.backend,
            rule_set="renju", seed=args.seed, config=cfg,
            metadata={
                "backends": backends,
                "components_used": cfg["components_used"],
                "source_checkpoint": str(source.resolve()),
            },
        )
    from az.backend_logging import log_backend_selection
    log_backend_selection(
        run_id=run_id,
        workload="selfplay",
        requested_backend=args.backend,
        backends=backends,
        components_used=cfg["components_used"],
        primary_path=db_path,
    )
    rules = get_rules("renju")
    buffer = ReplayBuffer(cfg["buffer_size"])
    trainer = Trainer.from_pretrained(source, cfg, buffer=buffer, seed=args.seed)

    for iteration in range(cfg["n_iter"]):
        records = list(run_games(
            trainer.model, rules, cfg, cfg["games_per_iter"], seed=args.seed + iteration,
            buffer=buffer, model_iter=iteration,
        ))
        with db:
            for record in records:
                logging_v2.record_selfplay_game(db, run_id, {
                    "game_id": iteration * int(cfg["games_per_iter"]) + record.game_id,
                    "timestamp": record.timestamp,
                    "rule_set": record.rule_set,
                    "first_player": record.first_player,
                    "second_player": record.second_player,
                    "winner": record.winner,
                    "first_player_won": record.first_player_won,
                    "end_reason": record.end_reason,
                    "moves_count": record.moves_count,
                    "duration_sec": record.duration_sec,
                    "nn_calls": record.nn_calls,
                    "mean_batch": record.mean_batch,
                    "model_iter": record.model_iter,
                    "config": record.config,
                    "moves": [vars(move) for move in record.moves] if record.moves else None,
                    "override_count": record.override_count,
                    "override_rate": record.override_rate,
                    "root_data": record.root_data,
                })
        if trainer.eval_batch is None:
            trainer.freeze_eval_batch()
        stats = trainer.train_steps()
        eval_loss = trainer.loss_on_fixed_batch()
        trainer.n_iter = iteration + 1
        trainer.save(checkpoint)
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
            "backends": backends,
            "components_used": cfg["components_used"],
            "global_step": trainer.global_step,
        }
        with db:
            logging_v2.record_training_step(
                db,
                run_id=run_id,
                iteration=trainer.n_iter,
                global_step=trainer.global_step,
                recorded_at=datetime.now(timezone.utc).isoformat(),
                metrics=row,
            )
        print(json.dumps(row))
    db.close()
    print(f"done: {trainer.n_iter} iterations; checkpoint={checkpoint}; database={db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
