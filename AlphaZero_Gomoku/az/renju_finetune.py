from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch

from az import generation_retention, training_recovery
from az.backend import configure_backends
from az.logging_v2 import (
    connect, initialize, json_dumps, record_selfplay_game, record_selfplay_run,
    record_training_step, write_event,
)
from az.replay_buffer import ReplayBuffer
from az.trainer import Trainer, resolve_generation
from az.selfplay import run_games, run_games_parallel
from az.training_health import outcome_counts, outcome_imbalance_message
from config import load_config
from gomoku.rules import get_rules

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_or_nested(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        try:
            right.relative_to(left)
            return True
        except ValueError:
            return False


def _validate_paths(out_dir: Path, db_path: Path) -> None:
    out = out_dir.resolve()
    db = db_path.resolve()
    normal_checkpoint = Path("checkpoints/normal").resolve()
    normal_db = Path("data/normal.sqlite").resolve()
    if _same_or_nested(out, normal_checkpoint) or db == normal_db or _same_or_nested(db, normal_checkpoint):
        raise ValueError("Renju output paths must not overlap Normal artifacts.")
    if out == db or _same_or_nested(out, db):
        raise ValueError("Renju checkpoint and database paths must not overlap.")
    if out.exists():
        if out.is_symlink() or not out.is_dir():
            raise ValueError("Renju output path is not a safe directory.")
        allowed = {"latest.json", "generations", "recovery"}
        unexpected = {path.name for path in out.iterdir()} - allowed
        if unexpected:
            raise ValueError(f"Renju output directory contains unexpected entries: {sorted(unexpected)}")
        if (out / "generations").exists() and ((out / "generations").is_symlink()
                or not (out / "generations").is_dir()):
            raise ValueError("Renju generations path is not a safe directory.")
        if (out / "recovery").exists() and ((out / "recovery").is_symlink()
                or not (out / "recovery").is_dir()):
            raise ValueError("Renju recovery path is not a safe directory.")


def _run_row(db, run_id, started_at, backend, seed, config, metadata, resume, allow_extend=False,
             allow_execution_change=False):
    row = db.execute(
        "SELECT started_at, backend, rule_set, board_size, seed, config_json, metadata_json "
        "FROM runs WHERE run_id=?", (run_id,)
    ).fetchone()
    expected = (started_at, backend, "renju", 11, seed, json_dumps(config), json_dumps(metadata))
    if row is None:
        if resume:
            raise ValueError("Cannot resume: Renju run metadata is missing.")
        record_selfplay_run(db, run_id=run_id, started_at=started_at, backend=backend,
                            rule_set="renju", seed=seed, config=config, metadata=metadata)
    elif (tuple(row)[1:5] != expected[1:5]
          or (not allow_extend and row[5] != expected[5])
          or (not allow_execution_change and row[6] != expected[6])):
        raise ValueError("Cannot resume: Renju run identity or configuration differs.")


def _record_games(db, run_id, records):
    with db:
        for record in records:
            record_selfplay_game(db, run_id, {
                "game_id": record.game_id,
                "timestamp": record.timestamp, "rule_set": record.rule_set,
                "first_player": record.first_player, "second_player": record.second_player,
                "winner": record.winner, "first_player_won": record.first_player_won,
                "end_reason": record.end_reason, "moves_count": record.moves_count,
                "duration_sec": record.duration_sec, "nn_calls": record.nn_calls,
                "mean_batch": record.mean_batch, "model_iter": record.model_iter,
                "config": record.config,
                "moves": [vars(move) for move in record.moves] if record.moves else None,
                "override_count": record.override_count, "override_rate": record.override_rate,
                "root_data": record.root_data,
            })


def _stop_if_degenerate(db, run_id, cfg) -> bool:
    """Return True and log an event when the run's outcome distribution is one-sided.

    A degenerate distribution teaches a colour shortcut instead of a position value,
    so the run stops and stays resumable rather than spending its remaining budget.
    """
    counts = outcome_counts(db, run_id)
    message = outcome_imbalance_message(
        counts,
        min_games=int(cfg.get("outcome_guard_min_games", 0)),
        max_side_win_rate=float(cfg.get("outcome_guard_max_side_win_rate", 1.01)),
    )
    if message is None:
        return False
    already_logged = db.execute(
        "SELECT 1 FROM events WHERE run_id=? AND event='outcome_guard_stop' LIMIT 1", (run_id,)
    ).fetchone()
    if already_logged is None:
        with db:
            write_event(
                db, event="outcome_guard_stop", level="warning", run_id=run_id,
                payload={"reason": message, "counts": counts},
                recorded_at=datetime.now(timezone.utc).isoformat(),
            )
    print(f"stopped: {message}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune the official Normal checkpoint on Renju self-play.")
    parser.add_argument("--config", default="configs/renju.yaml")
    parser.add_argument("--source", default="checkpoints/normal/latest.json")
    parser.add_argument("--out-dir", default="checkpoints/renju_finetune_1500")
    parser.add_argument("--db", default="data/renju_finetune_1500.sqlite")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-games", type=int)
    parser.add_argument("--games", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--worker-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-iter-target", type=int, help="Absolute target iteration for resume schedule extension.")
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--mcts-backend", choices=("python", "cpp"))
    parser.add_argument("--patterns-backend", choices=("python", "cpp"))
    args = parser.parse_args(argv)
    if args.iterations is not None and args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.max_new_games is not None and args.max_new_games <= 0:
        raise SystemExit("--max-new-games must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.cpu_threads <= 0:
        raise SystemExit("--cpu-threads must be positive")
    if args.n_iter_target is not None and args.n_iter_target <= 0:
        raise SystemExit("--n-iter-target must be positive")

    cfg = load_config(args.config)
    if cfg["rule_set"] != "renju":
        raise SystemExit("fine-tuning config must set rule_set: renju")
    if args.iterations is not None:
        cfg["n_iter"] = args.iterations
    if args.games is not None:
        cfg["games_per_iter"] = args.games
    out_dir, db_path = Path(args.out_dir), Path(args.db)
    _validate_paths(out_dir, db_path)
    source_pointer = Path(args.source)
    if not source_pointer.is_file():
        raise SystemExit(f"source checkpoint not found: {source_pointer}")
    try:
        source_checkpoint, _ = resolve_generation(source_pointer)
    except (ValueError, OSError, KeyError) as exc:
        raise SystemExit(f"source must be a validated generation pointer: {exc}") from exc
    source_meta = torch.load(source_checkpoint, map_location="cpu", weights_only=False).get("meta", {})
    source_generation = source_checkpoint.parent.name
    source_sha = _sha256(source_checkpoint)
    source_identity = {"checkpoint_sha256": source_sha, "generation": source_generation}
    if source_meta.get("rule_set") != "normal" or source_meta.get("n_iter") != 1500:
        raise SystemExit("source must be a Normal checkpoint at iteration 1500")

    pointer = out_dir / "latest.json"
    if args.resume:
        if not pointer.is_file():
            raise SystemExit("--resume requires an existing Renju generation")
        checkpoint, replay_path = resolve_generation(pointer)
        saved_meta = torch.load(checkpoint, map_location="cpu", weights_only=False).get("meta", {})
        run_id = args.run_id or saved_meta.get("run_id")
        if not run_id or saved_meta.get("bootstrap", {}).get("source") != source_identity:
            raise ValueError("Cannot resume: source checkpoint identity differs.")
        saved_cfg = saved_meta.get("config")
        if not isinstance(saved_cfg, dict):
            raise ValueError("Cannot resume: checkpoint config is missing.")
        for key in ("rule_set", "n_blocks", "n_filters", "learning_rate", "weight_decay",
                    "lr_schedule", "train_steps_per_iter", "games_per_iter",
                    "selfplay_opening_plies", "parallel_games", "n_simulations",
                    "buffer_size", "batch_size", "pi_target_temp", "temp_moves",
                    "dirichlet_alpha", "dirichlet_eps", "c_puct"):
            if saved_cfg.get(key) != cfg.get(key):
                raise ValueError("Cannot resume: Renju training configuration differs.")
        if args.n_iter_target is not None and args.n_iter_target < int(saved_cfg["n_iter"]):
            raise ValueError("--n-iter-target cannot reduce the saved schedule horizon.")
        cfg = {**saved_cfg, "n_iter": max(int(saved_cfg["n_iter"]), int(args.n_iter_target or saved_cfg["n_iter"]))}
        buffer = ReplayBuffer.load_npz(replay_path)
        trainer = Trainer.load(checkpoint, {**cfg, "n_iter": int(saved_cfg["n_iter"])},
                               buffer=buffer, seed=args.seed)
        if trainer.run_id != run_id or not isinstance(trainer.bootstrap, dict):
            raise ValueError("Cannot resume: run identity differs.")
        started_at = trainer.bootstrap["run_started_at"]
    else:
        if pointer.exists():
            raise SystemExit("Renju output already has a committed generation; use --resume")
        run_id = args.run_id or f"renju-finetune-{uuid.uuid4().hex}"
        started_at = datetime.now(timezone.utc).isoformat()
        buffer = ReplayBuffer(cfg["buffer_size"])
        trainer = Trainer.from_pretrained(source_checkpoint, cfg, buffer=buffer, seed=args.seed)
        trainer.run_id = run_id
        trainer.bootstrap = {"source": source_identity, "run_started_at": started_at}

    try:
        backends = configure_backends(
            args.backend, mcts_backend=args.mcts_backend,
            patterns_backend=args.patterns_backend or ("python" if args.backend == "auto" else None),
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if args.resume:
        cfg = {**cfg, "n_iter": max(int(saved_cfg["n_iter"]), int(args.n_iter_target or saved_cfg["n_iter"]))}
    cfg = {**cfg, "backend": args.backend, "workers": args.workers,
           "worker_device": args.worker_device,
           "backend_metadata": backends,
           "mcts_backend": backends["mcts"], "patterns_backend": backends["patterns"],
           "components_used": {"mcts": backends["mcts"], "patterns": "not_used"},
           "provenance": {"run_id": run_id, "source_checkpoint_sha256": source_sha,
                          "source_generation": source_generation}}
    if args.resume and any(saved_cfg.get(key) != cfg.get(key) for key in
                           ("backend", "mcts_backend", "patterns_backend")) and args.iterations is None and args.max_new_games is None and args.n_iter_target is None:
        raise ValueError("Cannot resume: selected backend differs from the saved run.")
    trainer.cfg = cfg
    start_iteration = trainer.n_iter
    if args.workers > 1:
        if args.worker_device == "cuda" and not torch.cuda.is_available():
            raise SystemExit("--worker-device cuda requires an available CUDA device")
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(args.cpu_threads)
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=args.workers)
    else:
        pool = None
    if args.iterations is not None:
        target = max(int(cfg["n_iter"]), int(args.iterations)) if args.resume else start_iteration + args.iterations
        if args.resume and args.n_iter_target is not None:
            target = max(target, args.n_iter_target)
        cfg = {**cfg, "n_iter": target}
        trainer.cfg = cfg
    elif args.max_new_games is not None:
        target = start_iteration + args.max_new_games // int(cfg["games_per_iter"])
        cfg = {**cfg, "n_iter": max(target, int(args.n_iter_target or target))}
        trainer.cfg = cfg
    if args.resume and args.n_iter_target is None and args.iterations is None and args.max_new_games is None and cfg.get("backend") != args.backend:
        raise ValueError("Cannot resume: backend differs from the saved run.")
    if args.resume and args.n_iter_target is None and args.iterations is None and args.max_new_games is None and args.workers != int(saved_cfg.get("workers", 1)):
        raise ValueError("Cannot resume: worker count differs from the saved run.")
    if args.max_new_games is None:
        game_limit = (int(cfg["n_iter"]) - start_iteration) * int(cfg["games_per_iter"])
    else:
        game_limit = int(args.max_new_games)
    target_iteration = min(int(cfg["n_iter"]), start_iteration + game_limit // int(cfg["games_per_iter"]))

    initialize(db_path, "selfplay")
    db = connect(db_path)
    metadata = {"backends": backends, "source_checkpoint_sha256": source_sha,
                "source_generation": source_generation}
    try:
        _run_row(db, run_id, started_at, args.backend, args.seed, cfg, metadata, args.resume,
                 allow_extend=bool(args.resume and (args.iterations is not None or args.max_new_games is not None)),
                 allow_execution_change=bool(args.resume and (args.iterations is not None or args.max_new_games is not None)))
        db.commit()
        from az.backend_logging import log_backend_selection
        log_backend_selection(run_id=run_id, workload="selfplay", requested_backend=args.backend,
                              backends=backends, components_used=cfg["components_used"], primary_path=db_path)
        if not args.resume:
            trainer.save(pointer)
        rules = get_rules("renju")
        recovery = getattr(trainer, "recovery", None)
        if recovery and recovery["identity"]["run_id"] == run_id:
            identity = recovery["identity"]
            if identity["iteration"] + 1 != trainer.n_iter:
                raise ValueError("Recovery iteration does not match checkpoint")
            saved = training_recovery.load(out_dir, identity)
            if saved is None:
                raise ValueError("Committed checkpoint recovery outbox is missing")
            directory, _, recovered_records = saved
            if recovery.get("manifest_sha256") != _sha256(directory / "manifest.json"):
                raise ValueError("Recovery manifest hash mismatch")
            metrics = training_recovery.load_metrics(
                directory, recovery["metrics_name"], recovery["metrics_sha256"])
            _record_games(db, run_id, recovered_records)
            with db:
                record_training_step(
                    db, run_id=run_id, iteration=trainer.n_iter, global_step=trainer.global_step,
                    recorded_at=metrics["recorded_at"], metrics=metrics["row"])
        if _stop_if_degenerate(db, run_id, cfg):
            return 3
        while trainer.n_iter < target_iteration:
            iteration = trainer.n_iter
            iteration_source_sha = _sha256(trainer._generation_checkpoint) if trainer._generation_checkpoint else source_sha
            identity = training_recovery._identity(run_id, iteration, iteration_source_sha)
            training_recovery.reject_other_source(out_dir, identity)
            saved = training_recovery.load(out_dir, identity)
            if saved is not None:
                directory, buffer, records = saved
                trainer.buffer = buffer
            else:
                if db.execute("SELECT 1 FROM games WHERE run_id=? AND json_extract(payload_json, '$.model_iter')=?",
                              (run_id, iteration)).fetchone():
                    raise ValueError("Existing Renju games have no recovery outbox; refusing to regenerate.")
                capture = training_recovery.Capture(buffer, cfg["games_per_iter"])
                runtime_cfg = {**cfg, "_worker_cpu_threads": args.cpu_threads,
                               "_worker_device": args.worker_device,
                               "_game_offset": iteration * int(cfg["games_per_iter"]),
                               "provenance": {"run_id": run_id, "iteration": iteration,
                                              "source_checkpoint_sha256": iteration_source_sha}}
                try:
                    if pool is None:
                        records = list(run_games(trainer.model, rules, runtime_cfg, cfg["games_per_iter"],
                                                 seed=args.seed + iteration, buffer=capture, model_iter=iteration))
                    else:
                        worker_buffer = ReplayBuffer(cfg["buffer_size"])
                        records = run_games_parallel(pool, trainer.model, rules, runtime_cfg,
                                                     cfg["games_per_iter"], seed=args.seed + iteration,
                                                     buffer=worker_buffer, model_iter=iteration,
                                                     workers=args.workers)
                        for index in range(len(worker_buffer)):
                            capture.add(worker_buffer.features[index], worker_buffer.pi[index],
                                        worker_buffer.z[index], "renju",
                                        game_id=str(worker_buffer.game_ids[index]))
                        capture.samples = worker_buffer
                except BaseException:
                    if pool is not None:
                        pool.shutdown(wait=True, cancel_futures=True)
                    raise
                buffer = capture.buffer
                if sorted(record.game_id for record in records) != list(range(iteration * int(cfg["games_per_iter"]),
                                                                            (iteration + 1) * int(cfg["games_per_iter"]))):
                    raise ValueError("Self-play returned an incomplete or duplicate game-ID batch.")
                directory = training_recovery.prepare(out_dir, identity, buffer, records, capture.samples)
                trainer.buffer = buffer
            _record_games(db, run_id, records)
            if trainer.eval_batch is None:
                trainer.freeze_eval_batch()
            stats = trainer.train_steps()
            eval_loss = trainer.loss_on_fixed_batch()
            trainer.n_iter = iteration + 1
            row = {"iteration": trainer.n_iter, "run_id": run_id, "rule_set": "renju",
                   "source_checkpoint_sha256": source_sha, "source_generation": source_generation,
                   "games": len(records), "buffer_size": len(buffer), **stats,
                   "loss_eval_on_fixed_batch": eval_loss, "global_step": trainer.global_step,
                   "backends": backends, "components_used": cfg["components_used"],
                   "start_iteration": start_iteration, "game_limit": game_limit,
                   "target_iteration": target_iteration}
            recorded_at = datetime.now(timezone.utc).isoformat()
            metrics_name, digest = training_recovery.save_metrics(directory, row, recorded_at)
            trainer.recovery = {"identity": identity, "metrics_name": metrics_name,
                                "metrics_sha256": digest,
                                "manifest_sha256": _sha256(directory / "manifest.json")}
            trainer.bootstrap = {"source": source_identity, "run_started_at": started_at}
            trainer.save(pointer)
            with db:
                record_training_step(db, run_id=run_id, iteration=trainer.n_iter,
                                     global_step=trainer.global_step, recorded_at=recorded_at, metrics=row)
            generation_retention.apply(out_dir, dry_run=False)
            print(json.dumps(row, sort_keys=True))
            if _stop_if_degenerate(db, run_id, cfg):
                return 3
        if trainer.n_iter >= int(cfg["n_iter"]):
            print(f"done: {trainer.n_iter} iterations; checkpoint={pointer}; database={db_path}")
        else:
            print(f"paused: {trainer.n_iter} of {int(cfg['n_iter'])} iterations; checkpoint={pointer}; database={db_path}")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
