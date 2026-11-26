"""Continuously train the Normal az_v0 model and promote checkpoints by paired Elo."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import uuid
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.model import AZNet
from az.replay_buffer import ReplayBuffer
from az.selfplay import run_games, run_games_parallel
from az.trainer import Trainer, _write_pointer, resolve_generation
from az import generation_retention
from az import training_recovery
from config import load_config
from gomoku.rules.normal import NormalRules


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the AlphaZero Gomoku agent.")
    parser.add_argument("--config", default="configs/normal.yaml")
    parser.add_argument("--iterations", type=int, default=None, help="Absolute iteration target (resume defaults to one more).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--resume", action="store_true", help="Continue from a validated generation.")
    source.add_argument("--fresh", action="store_true", help="Start a new model in an empty output directory.")
    parser.add_argument("--generation", help="Explicit generation ID for rollback (requires --resume).")
    parser.add_argument("--out-dir", default="checkpoints/normal", help="Generation and pointer directory.")
    parser.add_argument("--db", default="data/normal.sqlite", help="Self-play/training SQLite database.")
    parser.add_argument("--tournament-db", default="data/tournament.sqlite", help="Promotion tournament SQLite database.")
    parser.add_argument("--run-id", help="Stable az_v0 run identity; reuse it when resuming that run.")
    parser.add_argument("--log-dir", help="Deprecated and ignored; use --db.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--games", type=int, default=None, help="Games per iteration (overrides cfg games_per_iter).")
    parser.add_argument("--simulations", type=int, default=None, help="MCTS simulations per move (overrides cfg).")
    parser.add_argument("--workers", type=int, default=1, help="Self-play processes; 1 keeps the single-process path.")
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default=None, help="Backend preset; auto uses C++ when available.")
    parser.add_argument("--mcts-backend", choices=("python", "cpp"), help="Override the MCTS backend preset.")
    parser.add_argument("--logging-level", choices=("none", "standard"), default="standard")
    parser.add_argument("--patterns-backend", choices=("python", "cpp"), help="Override the patterns backend preset.")
    return parser


def _resolve_run_id(requested: str | None, *, resume: bool, logging_level: str) -> str:
    if resume and logging_level == "STANDARD" and not requested:
        raise ValueError("STANDARD resume requires the original --run-id.")
    return requested or f"az_v0-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _ensure_training_run(
    db, logging_v2, *, run_id: str, started_at: str, backend: str,
    rule_set: str, seed: int, config: dict, metadata: dict, resume: bool,
) -> None:
    row = db.execute(
        "SELECT backend, rule_set, board_size, seed, config_json, metadata_json "
        "FROM runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    if row is None and resume:
        raise ValueError("STANDARD resume requires the original --run-id in the self-play database.")
    if row is not None and resume:
        expected = (
            backend, rule_set, 11, seed,
            logging_v2.json_dumps(config), logging_v2.json_dumps(metadata),
        )
        if tuple(row) != expected:
            raise ValueError(f"run_id already exists with different metadata: {run_id}")
        return
    logging_v2.record_selfplay_run(
        db, run_id=run_id, started_at=started_at, backend=backend,
        rule_set=rule_set, seed=seed, config=config, metadata=metadata,
    )


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _promote_generation(checkpoint: Path, best_pointer: Path) -> None:
    if best_pointer.is_symlink():
        raise ValueError("Best pointer must not be a symlink.")
    _write_pointer(best_pointer, checkpoint.parent.name)


def _run_promotion_pair(
    candidate: Path,
    opponent: str,
    opponent_checkpoint: Path | None,
    *,
    config_path: str,
    db_path: Path,
    run_id: str,
    games: int,
    seed: int,
    backends: dict | None = None,
    workers: int = 1,
) -> tuple[str, str, str]:
    from az.backend import configure_backends
    from az.elo import PROMOTION_EVAL_OPENING_PLIES
    from az.logging_v2 import connect, initialize
    from scripts.tournament import AgentSpec, Budget, Defaults, agent_settings, run_tournament
    if backends is None:
        backends = configure_backends("python", mcts_backend="python", patterns_backend="python")

    candidate_label = f"az_v0@checkpoint={candidate.resolve()}"
    opponent_label = (
        f"az_v0@checkpoint={opponent_checkpoint.resolve()}" if opponent_checkpoint else opponent
    )
    budget = Budget("sims", 200)
    entries = {
        candidate_label: agent_settings(
            AgentSpec(candidate_label, "az_v0", {"checkpoint": str(candidate)}), budget,
            Defaults(checkpoint=str(candidate), config=config_path),
        ),
        opponent_label: agent_settings(
            AgentSpec(opponent_label, opponent, {"checkpoint": str(opponent_checkpoint)} if opponent_checkpoint else {}),
            budget, Defaults(checkpoint=str(opponent_checkpoint) if opponent_checkpoint else None, config=config_path),
        ),
    }
    initialize(db_path, "tournament")
    connection = connect(db_path)
    try:
        resume = connection.execute(
            "SELECT 1 FROM experiments WHERE experiment_id=?", (run_id,)
        ).fetchone() is not None
    finally:
        connection.close()
    run_tournament(
        entries, games=games, budget=budget, ruleset="normal", seed=seed, run_id=run_id,
        workers=workers, opening_plies=PROMOTION_EVAL_OPENING_PLIES, torch_threads=1, resume=resume,
        pair=f"{candidate_label},{opponent_label}",
        backends=backends,
        db_path=db_path, logging_level="standard",
        extra_config={"model_id": "az_v0", "promotion_protocol": "az-v0-promotion-v1.4"},
    )
    return candidate_label, opponent_label, run_id


def _promotion_event(db, run_id: str, event: str, logging_v2):
    if db is None:
        return None
    row = db.execute(
        "SELECT payload_json FROM events WHERE run_id=? AND event=?", (run_id, event)
    ).fetchone()
    return json.loads(row["payload_json"]) if row else None

def _record_games(db, logging_v2, run_id, iteration, games_per_iter, records):
    with db:
        for record in records:
            record = replace(record, game_id=iteration * int(games_per_iter) + record.game_id)
            game_id = (record.config.get("provenance") or {}).get("game_id") or f"{run_id}:{record.game_id}"
            existing = db.execute(
                "SELECT run_id, local_game_id, rule_set, winner, end_reason, moves_count, duration_sec "
                "FROM games WHERE game_id=?", (game_id,)).fetchone()
            if existing is not None:
                if tuple(existing) != (run_id, str(record.game_id), record.rule_set, record.winner,
                                       record.end_reason, record.moves_count, record.duration_sec):
                    raise ValueError(f"game identity collision: {game_id}")
                for table, expected in (("moves", len(record.moves or ())), ("mcts_roots", len(record.root_data or ()))):
                    if db.execute(f"SELECT count(*) FROM {table} WHERE game_id=?", (game_id,)).fetchone()[0] != expected:
                        raise ValueError(f"game telemetry collision: {game_id}")
                for move in record.moves or ():
                    data = vars(move)
                    action = int(data["row"]) * 11 + int(data["col"])
                    saved_move = db.execute(
                        "SELECT player, action, latency_sec, selected_action, root_value, simulations, backend "
                        "FROM moves WHERE game_id=? AND move_index=?", (game_id, data["move_index"])).fetchone()
                    expected_move = (data["player"], action, data.get("latency_sec"), action,
                                     data.get("value_before"), data.get("simulations_run", data.get("simulations")),
                                     record.config.get("mcts_backend"))
                    if saved_move is None or tuple(saved_move) != expected_move:
                        raise ValueError(f"move identity collision: {game_id}/{data['move_index']}")
            logging_v2.record_selfplay_game(db, run_id, {
                "game_id": record.game_id, "timestamp": record.timestamp,
                "rule_set": record.rule_set, "first_player": record.first_player,
                "second_player": record.second_player, "winner": record.winner,
                "first_player_won": record.first_player_won, "end_reason": record.end_reason,
                "moves_count": record.moves_count, "duration_sec": record.duration_sec,
                "nn_calls": record.nn_calls, "mean_batch": record.mean_batch,
                "model_iter": record.model_iter, "config": record.config,
                "moves": [vars(move) for move in record.moves] if record.moves else None,
                "override_count": record.override_count, "override_rate": record.override_rate,
                "root_data": record.root_data,
            })


def _evaluate_promotion(
    trainer: Trainer,
    checkpoint_path: Path,
    best_path: Path,
    pointer_path: Path,
    *,
    config_path: str,
    tournament_db: Path,
    run_id: str,
    seed: int,
    logging_db,
    logging_v2,
    backends: dict | None = None,
    workers: int = 1,
) -> dict | None:
    from az.elo import (
        PROMOTION_EVAL_BOOTSTRAPS,
        PROMOTION_EVAL_GAMES_PER_PAIR,
        PROMOTION_EVAL_INTERVAL_GAMES,
        PROMOTION_EVAL_OPENING_PLIES,
        paired_davidson_ci,
    )

    current_games = trainer.n_iter * int(trainer.cfg["games_per_iter"])
    event_name = f"promotion_evaluation_{trainer.n_iter}"
    if current_games - trainer.promotion_eval_games >= PROMOTION_EVAL_INTERVAL_GAMES and (
        not checkpoint_path or not checkpoint_path.is_file()
    ):
        raise ValueError("Evaluation requires a valid candidate checkpoint.")
    best_pointer = best_path
    best_path = resolve_generation(best_pointer)[0] if best_pointer.is_file() else None
    saved = _promotion_event(logging_db, run_id, event_name, logging_v2)
    if saved is not None:
        candidate, _ = resolve_generation(pointer_path, saved["candidate_generation"])
        if saved.get("candidate_sha256") != _checkpoint_sha256(candidate):
            raise ValueError("Promotion event belongs to a different checkpoint.")
        if saved.get("promote") and (best_path is None or _checkpoint_sha256(candidate) != _checkpoint_sha256(best_path)):
            _promote_generation(candidate, best_pointer)
        if trainer.promotion_eval_games != current_games:
            trainer.promotion_eval_games = current_games
            trainer.save(pointer_path)
        return saved
    if current_games - trainer.promotion_eval_games < PROMOTION_EVAL_INTERVAL_GAMES:
        return None
    if current_games - trainer.promotion_eval_games != PROMOTION_EVAL_INTERVAL_GAMES:
        raise ValueError("Evaluation gate was crossed without a checkpoint at 5000 new games.")
    if logging_db is None:
        raise ValueError("automated best-checkpoint promotion requires STANDARD SQLite logging")
    candidate_sha = _checkpoint_sha256(checkpoint_path)
    candidate_generation = checkpoint_path.parent.name
    from az.logging_v2 import connect
    from scripts.tactical_benchmark import benchmark

    has_champion = best_path is not None
    base_seed = (seed + trainer.n_iter * 10000) & 0xFFFFFFFF
    threads = torch.get_num_threads()
    try:
        tactical = benchmark(str(checkpoint_path), "python", base_seed, split="holdout")
    finally:
        torch.set_num_threads(threads)
    if tactical["checkpoint_sha256"] != candidate_sha:
        raise ValueError("Tactical evaluation belongs to a different checkpoint.")
    candidate_label, opponent_label, experiment_id = _run_promotion_pair(
        checkpoint_path, "heuristic", None, config_path=config_path, db_path=tournament_db,
        run_id=f"{run_id}-promotion-{trainer.n_iter}-heuristic-0",
        games=PROMOTION_EVAL_GAMES_PER_PAIR, seed=base_seed,
        backends=backends, workers=workers,
    )
    connection = connect(tournament_db)
    try:
        comparison = paired_davidson_ci(
            connection, [experiment_id], candidate_label, opponent_label,
            bootstrap_samples=PROMOTION_EVAL_BOOTSTRAPS, seed=base_seed,
        )
    finally:
        connection.close()
    if comparison["games"] != PROMOTION_EVAL_GAMES_PER_PAIR:
        raise ValueError("Evaluation did not record exactly 100 paired games.")
    result = {
        "iteration": trainer.n_iter,
        "schedule_game_position": current_games,
        "candidate_sha256": candidate_sha,
        "best_sha256": _checkpoint_sha256(best_path) if has_champion else None,
        "comparisons": {"heuristic": comparison},
        "tactical": tactical,
        "evaluation_seed": base_seed,
        "experiment_id": experiment_id,
        "promotion_criterion": "paired 95% CI lower bound > 0 vs current best",
        "research_objective_met": None,
        "promote": not has_champion and comparison["ci95"][0] > 0,
        "status": "inconclusive_no_champion_comparison" if has_champion else "evaluated",
        "confirmed_regression": False,
        "protocol": "az-v0-promotion-v1.4",
        "evaluation_backends": backends,
        "evaluation_workers": workers,
        "opening_plies": PROMOTION_EVAL_OPENING_PLIES,
    }
    result["candidate_generation"] = candidate_generation
    if logging_db is not None:
        existing = logging_db.execute(
            "SELECT payload_json FROM events WHERE run_id=? AND event=?", (run_id, event_name)
        ).fetchone()
        if existing is not None:
            if json.loads(existing["payload_json"]) != result:
                raise ValueError(f"promotion event collision: {run_id}/{event_name}")
        else:
            with logging_db:
                logging_v2.write_event(
                    logging_db, run_id=run_id, event=event_name,
                    recorded_at=datetime.now(timezone.utc).isoformat(), payload=result,
                )
    trainer.promotion_eval_games = current_games
    trainer.save(pointer_path)
    if result.get("promote"):
        _promote_generation(checkpoint_path, best_pointer)
    return result


def main(argv: list[str] | None = None) -> int:

    args = build_parser().parse_args(argv)
    args.resume = not args.fresh
    if args.iterations is not None and args.iterations < 0:
        raise SystemExit("--iterations must be nonnegative")
    if args.generation and not args.resume:
        raise SystemExit("--generation requires --resume")
    pointer_path = Path(args.out_dir) / "latest.json"
    checkpoint_path = buffer_path = None
    if args.resume:
        checkpoint_path, buffer_path = resolve_generation(pointer_path, args.generation)
        source_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("meta", {})
        if not args.run_id and not args.generation:
            recovery = source_meta.get("recovery") or {}
            args.run_id = source_meta.get("run_id") or (recovery.get("identity") or {}).get("run_id")
        if args.seed is None and args.logging_level == "standard":
            owner = source_meta.get("run_id") or ((source_meta.get("recovery") or {}).get("identity") or {}).get("run_id")
            db_path = Path(args.db).resolve()
            if not db_path.is_file() or not owner:
                raise ValueError("Cannot resume: original run seed is unavailable.")
            with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as existing_db:
                row = existing_db.execute("SELECT seed FROM runs WHERE run_id=?", (owner,)).fetchone()
            if row is None:
                raise ValueError("Cannot resume: original run seed is unavailable.")
            args.seed = row[0]
        if args.backend is None:
            args.backend = source_meta.get("config", {}).get("backend", "auto")
    if args.seed is None:
        args.seed = 0
    if args.backend is None:
        args.backend = "auto"
    try:
        run_id = _resolve_run_id(args.run_id, resume=args.resume, logging_level=args.logging_level.upper())
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    from az.backend import configure_backends
    patterns_backend = args.patterns_backend
    mcts_backend = args.mcts_backend
    if args.resume:
        saved_config = source_meta.get("config", {})
        patterns_backend = patterns_backend or saved_config.get("patterns_backend")
        mcts_backend = mcts_backend or saved_config.get("mcts_backend")
    if args.backend == "auto" and patterns_backend is None:
        patterns_backend = "python"
    try:
        backends = configure_backends(args.backend, mcts_backend=mcts_backend,
                                      patterns_backend=patterns_backend)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"backends": backends}, sort_keys=True))
    cfg = load_config(args.config)
    if args.resume:
        source_cfg = source_meta.get("config")
        if not isinstance(source_cfg, dict):
            raise ValueError("Cannot resume: checkpoint configuration is missing.")
        cfg = dict(source_cfg)
    cfg["backend"] = args.backend
    cfg["backend_metadata"] = backends
    cfg["patterns_backend"] = backends["patterns"]
    cfg["mcts_backend"] = backends["mcts"]
    logging_level = args.logging_level.upper()
    cfg["model_id"] = "az_v0"
    cfg["components_used"] = {"mcts": backends["mcts"], "patterns": "not_used"}
    if args.iterations is not None and not args.resume:
        cfg["n_iter"] = args.iterations
    if args.games is not None:
        cfg["games_per_iter"] = args.games
    if args.simulations is not None:
        cfg["n_simulations"] = args.simulations
    if args.resume and args.games is not None and args.games != source_cfg.get("games_per_iter"):
        raise SystemExit("Changing games_per_iter on resume would invalidate the promotion counter.")
    from az.elo import PROMOTION_EVAL_INTERVAL_GAMES
    if (int(cfg["n_iter"]) * int(cfg["games_per_iter"]) >= PROMOTION_EVAL_INTERVAL_GAMES
            and PROMOTION_EVAL_INTERVAL_GAMES % int(cfg["games_per_iter"])):
        raise SystemExit("games_per_iter must divide 5000 when the run reaches an evaluation gate")
    if cfg["rule_set"] != "normal":
        raise SystemExit(f"rule_set '{cfg['rule_set']}' has no rules implementation yet (T24).")
    run_config = {**cfg, "logging_level": logging_level}
    runtime_cfg = {**cfg, "mcts_logging_level": logging_level}

    rules = NormalRules()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = out_dir / "latest.json"
    best_path = out_dir / "best.json"
    tournament_db_path = Path(args.tournament_db)
    if not args.resume and any((out_dir / name).exists() for name in ("latest.json", "best.json", "latest.pt", "buffer_latest.npz", "best.pt")):
        raise SystemExit(f"{out_dir} already contains checkpoints; use a new directory or --resume")
    if args.resume:
        if checkpoint_path is None or buffer_path is None:
            raise SystemExit("--resume needs a validated generation")
        buffer = ReplayBuffer.load_npz(buffer_path)
        trainer = Trainer.load(checkpoint_path, cfg, buffer=buffer, seed=args.seed)
        end_iteration = args.iterations if args.iterations is not None else trainer.n_iter + 1
        if end_iteration < trainer.n_iter:
            raise ValueError("Iteration target precedes the resumed checkpoint.")
        if end_iteration > cfg["n_iter"] and cfg.get("lr_schedule") == "cosine":
            raise ValueError("Cannot extend the checkpoint's cosine schedule horizon.")
        if trainer.bootstrap and args.logging_level != "standard":
            raise SystemExit("bootstrap continuation requires STANDARD logging for game provenance")
        recovery = getattr(trainer, "recovery", None)
        recorded_run_id = trainer.run_id or (recovery["identity"]["run_id"] if recovery else None)
        if args.logging_level == "standard" and not args.generation and recorded_run_id != run_id:
            raise ValueError("STANDARD resume requires the original --run-id; use --generation for rollback.")
        trainer.run_id = run_id
        print(f"resumed at iteration {trainer.n_iter}, global_step {trainer.global_step}")
    else:
        buffer = ReplayBuffer(cfg["buffer_size"])
        trainer = Trainer(AZNet(n_blocks=cfg["n_blocks"], n_filters=cfg["n_filters"]), cfg, buffer=buffer, seed=args.seed)
        end_iteration = cfg["n_iter"]
        trainer.run_id = run_id
        checkpoint_path = trainer.save(pointer_path)
    from az import logging_v2
    db_path = Path(args.db)
    db = None
    if args.logging_level == "standard":
        db_path.parent.mkdir(parents=True, exist_ok=True)
        logging_v2.initialize(db_path, "selfplay")
        db = logging_v2.connect(db_path)
        metadata = {"backends": backends, "components_used": cfg["components_used"]}
        existing = db.execute(
            "SELECT metadata_json FROM runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if existing is not None and args.resume:
            metadata = json.loads(existing["metadata_json"])
            if args.generation:
                expected = {"generation": args.generation, "checkpoint_sha256": _checkpoint_sha256(checkpoint_path)}
                if metadata.get("rollback_source") != expected:
                    raise ValueError("Explicit rollback requires a new --run-id or the same rollback source.")
                latest, _ = resolve_generation(pointer_path)
                latest_meta = torch.load(latest, map_location="cpu", weights_only=False).get("meta", {})
                if (latest_meta.get("run_id") == run_id or db.execute(
                        "SELECT 1 FROM training_steps WHERE run_id=? LIMIT 1", (run_id,)).fetchone()):
                    raise ValueError("Rollback already committed; use --resume without --generation.")
        else:
            metadata["parent_checkpoint_sha256"] = (
                trainer.bootstrap["parent_checkpoint_sha256"] if trainer.bootstrap else
                (_checkpoint_sha256(checkpoint_path) if args.resume else None)
            )
            metadata["bootstrap"] = trainer.bootstrap
            if args.generation:
                metadata["rollback_source"] = {
                    "generation": args.generation, "checkpoint_sha256": _checkpoint_sha256(checkpoint_path)}
        _ensure_training_run(
            db, logging_v2, run_id=run_id, started_at=datetime.now(timezone.utc).isoformat(),
            backend=args.backend, rule_set=cfg["rule_set"], seed=args.seed, config=run_config,
            metadata=metadata, resume=bool(args.resume and (not args.generation or existing is not None)),
        )
        db.commit()
        from az.backend_logging import log_backend_selection
        log_backend_selection(
            run_id=run_id, workload="selfplay", requested_backend=args.backend,
            backends=backends, components_used=cfg["components_used"], primary_path=db_path,
        )

    recovery = getattr(trainer, "recovery", None)
    if db is not None and recovery and recovery["identity"]["run_id"] == run_id:
        identity = recovery["identity"]
        if identity["iteration"] + 1 != trainer.n_iter:
            raise ValueError("Recovery iteration does not match checkpoint")
        saved = training_recovery.load(out_dir, identity)
        if saved is None:
            raise ValueError("Committed checkpoint recovery outbox is missing")
        directory, _, recovered_records = saved
        if recovery.get("manifest_sha256") != _checkpoint_sha256(directory / "manifest.json"):
            raise ValueError("Recovery manifest hash mismatch")
        metrics = training_recovery.load_metrics(directory, recovery["metrics_name"], recovery["metrics_sha256"])
        _record_games(db, logging_v2, run_id, identity["iteration"], cfg["games_per_iter"], recovered_records)
        with db:
            logging_v2.record_training_step(
                db, run_id=run_id, iteration=trainer.n_iter, global_step=trainer.global_step,
                recorded_at=metrics["recorded_at"], metrics=metrics["row"])


    startup_evaluation = _evaluate_promotion(
        trainer, checkpoint_path, best_path, pointer_path, config_path=args.config,
        tournament_db=tournament_db_path, run_id=run_id, seed=args.seed,
        logging_db=db, logging_v2=logging_v2, backends=backends, workers=args.workers,
    )
    if startup_evaluation is not None:
        checkpoint_path, _ = resolve_generation(pointer_path)
    if startup_evaluation:
        print(json.dumps({"promotion_evaluation": startup_evaluation}, sort_keys=True))
    if startup_evaluation and startup_evaluation.get("confirmed_regression"):
        print("confirmed quality regression; stopping this training session, best checkpoint retained")
        if db is not None:
            db.close()
        return 0
    device_name = torch.cuda.get_device_name(0) if trainer.device.type == "cuda" else "cpu"
    print(f"device: {trainer.device.type} ({device_name})")

    if args.workers > 1:
        print(f"self-play workers: {args.workers}")

    for iteration in range(trainer.n_iter, end_iteration):
        runtime_cfg["provenance"] = {
            "run_id": run_id, "iteration": iteration,
            "source_checkpoint_sha256": _checkpoint_sha256(checkpoint_path),
        }
        identity = training_recovery._identity(run_id, iteration, runtime_cfg["provenance"]["source_checkpoint_sha256"])
        if db is not None:
            training_recovery.reject_other_source(out_dir, identity)
        saved = training_recovery.load(out_dir, identity) if db is not None else None
        selfplay_started = time.perf_counter()
        if saved is not None:
            directory, buffer, records = saved
            trainer.buffer = buffer
        else:
            if db is not None:
                for existing_game in db.execute("SELECT payload_json FROM games WHERE run_id=?", (run_id,)):
                    payload = json.loads(existing_game[0])
                    if payload.get("model_iter") == iteration:
                        raise ValueError("Existing games lack a matching recovery outbox; refusing to regenerate")
            capture = training_recovery.Capture(buffer, cfg["games_per_iter"]) if db is not None else None
            target_buffer = capture if capture is not None else buffer
            if args.workers <= 1:
                records = list(run_games(
                    trainer.model, rules, runtime_cfg, runtime_cfg["games_per_iter"],
                    seed=args.seed + iteration, buffer=target_buffer, model_iter=iteration,
                ))
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as executor:
                    records = run_games_parallel(
                        executor, trainer.model, rules, runtime_cfg, runtime_cfg["games_per_iter"],
                        seed=args.seed + iteration, buffer=target_buffer, model_iter=iteration,
                        workers=args.workers,
                    )
            if db is not None:
                directory = training_recovery.prepare(out_dir, identity, buffer, records, capture.samples)
        selfplay_sec = time.perf_counter() - selfplay_started
        if db is not None:
            _record_games(db, logging_v2, run_id, iteration, cfg["games_per_iter"], records)

        if trainer.eval_batch is None and len(buffer) > 0:
            trainer.freeze_eval_batch()

        train_started = time.perf_counter()
        stats = trainer.train_steps()
        train_sec = time.perf_counter() - train_started
        eval_loss = trainer.loss_on_fixed_batch()

        trainer.n_iter = iteration + 1
        promotion = None
        games = len(records)
        parent_sha = runtime_cfg["provenance"]["source_checkpoint_sha256"]
        if trainer.lineage is None:
            trainer.lineage = {"historical_games": None, "generated_since_tracking": 0,
                               "identified_game_ids_used_since_tracking": [],
                               "identified_game_iteration_uses": 0,
                               "unidentified_sample_iteration_uses": 0}
        used_ids = set(trainer.lineage["identified_game_ids_used_since_tracking"])
        used_ids.update(stats["identified_game_ids_used"])
        trainer.lineage = {**trainer.lineage,
                           "generated_since_tracking": trainer.lineage["generated_since_tracking"] + games,
                           "identified_game_ids_used_since_tracking": sorted(used_ids),
                           "identified_game_iteration_uses": trainer.lineage["identified_game_iteration_uses"] + stats["identified_games_used"],
                           "unidentified_sample_iteration_uses": trainer.lineage["unidentified_sample_iteration_uses"] + stats["unidentified_samples_used"]}
        row = {
            "iteration": trainer.n_iter,
            "device": device_name,
            "selfplay_sec": round(selfplay_sec, 2),
            "train_sec": round(train_sec, 2),
            "games": games,
            "buffer_size": len(buffer),
            "identified_games_used": stats["identified_games_used"],
            "unidentified_samples_used": stats["unidentified_samples_used"],
            "parent_checkpoint_sha256": parent_sha,
            "parent_generation": checkpoint_path.parent.name,
            "run_id": run_id,
            "lineage": {key: value for key, value in trainer.lineage.items()
                        if key != "identified_game_ids_used_since_tracking"}
                       | {"identified_unique_games_used_since_tracking": len(used_ids)},
            "distinct_samples": stats["distinct_samples"],
            "loss": round(stats["loss"], 5),
            "policy_loss": round(stats["policy_loss"], 5),
            "value_loss": round(stats["value_loss"], 5),
            "loss_eval_on_fixed_batch": round(eval_loss, 5),
            "mean_moves_per_game": round(sum(r.moves_count for r in records) / games, 2),
            "nn_calls_per_game": round(sum(r.nn_calls for r in records) / games, 1),
            "mean_batch": round(sum(r.mean_batch for r in records) / games, 3),
            "global_step": trainer.global_step,
            "backends": backends,
            "components_used": {"mcts": backends["mcts"], "patterns": "not_used"},
        }
        if db is not None:
            recorded_at = datetime.now(timezone.utc).isoformat()
            metrics_name, digest = training_recovery.save_metrics(directory, row, recorded_at)
            metrics = {"row": row, "recorded_at": recorded_at}
            trainer.recovery = {"identity": identity, "metrics_name": metrics_name, "metrics_sha256": digest,
                                "manifest_sha256": _checkpoint_sha256(directory / "manifest.json")}
        else:
            trainer.recovery = None
        checkpoint_path = trainer.save(pointer_path)
        if db is not None:
            with db:
                logging_v2.record_training_step(
                    db, run_id=run_id, iteration=trainer.n_iter, global_step=trainer.global_step,
                    recorded_at=metrics["recorded_at"], metrics=row,
                )
        # Conservative retention: only delete after the new generation is committed.
        try:
            retention = generation_retention.apply(out_dir, dry_run=False)
            if retention.get("deleted"):
                print(json.dumps({"retention": retention}, sort_keys=True))
        except Exception as exc:  # fail closed: never affect training on cleanup error
            print(json.dumps({"retention_error": str(exc)}))
        promotion = _evaluate_promotion(
            trainer, checkpoint_path, best_path, pointer_path, config_path=args.config,
            tournament_db=tournament_db_path, run_id=run_id, seed=args.seed,
            logging_db=db, logging_v2=logging_v2, backends=backends, workers=args.workers,
        )
        if promotion is not None:
            checkpoint_path, _ = resolve_generation(pointer_path)
            row["promotion_evaluation"] = promotion
        print(json.dumps(row))
        if promotion and promotion.get("confirmed_regression"):
            print("confirmed quality regression; stopping this training session, best checkpoint retained")
            break
    if db is not None:
        db.close()
    database_status = f"database {db_path}" if db_path and db is not None else "logging NONE"
    print(f"done: {trainer.n_iter} iterations, latest {pointer_path}, best {best_path}, run {run_id}, {database_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
