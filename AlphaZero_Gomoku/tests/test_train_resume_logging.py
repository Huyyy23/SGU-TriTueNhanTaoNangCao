import pytest

from az import logging_v2
from scripts.train import _ensure_training_run, _resolve_run_id


def test_standard_resume_requires_the_original_run_id():
    with pytest.raises(ValueError, match="requires the original --run-id"):
        _resolve_run_id(None, resume=True, logging_level="STANDARD")
    assert _resolve_run_id("run-1", resume=True, logging_level="STANDARD") == "run-1"
    assert _resolve_run_id(None, resume=True, logging_level="NONE").startswith("az_v0-")


def test_resume_reuses_existing_run_and_rejects_metadata_drift(tmp_path):
    database = tmp_path / "selfplay.sqlite"
    logging_v2.initialize(database, "selfplay")
    config = {"n_iter": 4, "logging_level": "STANDARD"}
    metadata = {"backends": {"mcts": "python"}}
    with logging_v2.connect(database) as db:
        _ensure_training_run(
            db, logging_v2, run_id="run-1", started_at="2026-01-01T00:00:00Z",
            backend="python", rule_set="normal", seed=7, config=config,
            metadata=metadata, resume=False,
        )
        db.commit()

        _ensure_training_run(
            db, logging_v2, run_id="run-1", started_at="2026-01-02T00:00:00Z",
            backend="python", rule_set="normal", seed=7, config=config,
            metadata=metadata, resume=True,
        )
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
        assert db.execute("SELECT started_at FROM runs WHERE run_id='run-1'").fetchone()[0] == "2026-01-01T00:00:00Z"

        with pytest.raises(ValueError, match="different metadata"):
            _ensure_training_run(
                db, logging_v2, run_id="run-1", started_at="2026-01-02T00:00:00Z",
                backend="python", rule_set="normal", seed=7,
                config={**config, "n_iter": 5}, metadata=metadata, resume=True,
            )


@pytest.mark.parametrize("has_best", [False, True])
def test_promotion_waits_5000_games_and_plays_100_total(tmp_path, monkeypatch, has_best):
    from types import SimpleNamespace

    from az import elo
    from scripts import train

    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"candidate")
    best = tmp_path / "best.json"
    champion = tmp_path / "champion.pt"
    if has_best:
        best.write_text("{}")
        champion.write_bytes(b"champion")
        monkeypatch.setattr(train, "resolve_generation", lambda *args: (checkpoint if len(args) > 1 else champion, None))
    else:
        monkeypatch.setattr(train, "resolve_generation", lambda *args: (checkpoint, None))
    monkeypatch.setattr(train, "_promote_generation", lambda *args: None)
    calls = []

    def run_pair(candidate, opponent, opponent_checkpoint, **kwargs):
        calls.append((opponent, opponent_checkpoint, kwargs["games"], kwargs["backends"], kwargs["workers"]))
        return "candidate", "opponent", kwargs["run_id"]

    monkeypatch.setattr(train, "_run_promotion_pair", run_pair)
    monkeypatch.setattr("scripts.tactical_benchmark.benchmark", lambda path, backend, seed, split: {
        "checkpoint_sha256": train._checkpoint_sha256(checkpoint),
        "accuracy": {"overall": {"correct": 12, "count": 24, "accuracy": 0.5}},
    })
    monkeypatch.setattr(elo, "paired_davidson_ci", lambda *args, **kwargs: {"ci95": [1, 2], "games": 100})
    trainer = SimpleNamespace(
        n_iter=599, cfg={"games_per_iter": 50}, promotion_eval_games=25000,
        save=lambda *args: None,
    )
    args = (trainer, checkpoint, best, tmp_path / "buffer.npz")
    kwargs = dict(config_path="normal.yaml", tournament_db=tmp_path / "tournament.sqlite",
                  run_id="test", seed=0, logging_v2=logging_v2,
                  backends={"requested_backend": "cpp", "mcts": "cpp", "patterns": "cpp"},
                  workers=2)
    assert train._evaluate_promotion(*args, logging_db=None, **kwargs) is None
    assert not calls

    database = tmp_path / "selfplay.sqlite"
    logging_v2.initialize(database, "selfplay")
    with logging_v2.connect(database) as db:
        _ensure_training_run(
            db, logging_v2, run_id="test", started_at="2026-01-01T00:00:00Z",
            backend="python", rule_set="normal", seed=0, config={}, metadata={}, resume=False,
        )
        trainer.n_iter = 600
        result = train._evaluate_promotion(*args, logging_db=db, **kwargs)
        assert train._evaluate_promotion(*args, logging_db=db, **kwargs) == result
        assert db.execute("SELECT count(*) FROM events WHERE event=?", ("promotion_evaluation_600",)).fetchone()[0] == 1
    assert calls == [("heuristic", None, 100, kwargs["backends"], 2)]
    assert result["promote"] is (not has_best)
    assert result["tactical"]["accuracy"]["overall"]["correct"] == 12
    assert result["comparisons"]["heuristic"]["ci95"] == [1, 2]
    assert result["research_objective_met"] is None
    assert trainer.promotion_eval_games == 30000



def test_promotion_tournament_uses_selected_cpp_backend_and_workers(tmp_path, monkeypatch):
    from scripts import train, tournament

    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"checkpoint")
    backends = {"requested_backend": "cpp", "mcts": "cpp", "patterns": "cpp"}
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return tmp_path / "tournament.sqlite", {}

    monkeypatch.setattr(tournament, "run_tournament", capture)
    train._run_promotion_pair(
        candidate, "heuristic", None, config_path="configs/normal.yaml",
        db_path=tmp_path / "tournament.sqlite", run_id="gate", games=100, seed=7,
        backends=backends, workers=2,
    )
    assert seen["backends"] == backends
    assert seen["workers"] == 2
    assert seen["extra_config"]["promotion_protocol"] == "az-v0-promotion-v1.4"

def test_gate_propagates_tactical_failure_without_advancing(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import train

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    trainer = SimpleNamespace(n_iter=100, cfg={"games_per_iter": 50},
                              promotion_eval_games=0, save=lambda *args: None)
    database = tmp_path / "selfplay.sqlite"
    logging_v2.initialize(database, "selfplay")
    def fail(*args, **kwargs):
        raise RuntimeError("tactical failed")
    monkeypatch.setattr("scripts.tactical_benchmark.benchmark", fail)
    with logging_v2.connect(database) as db:
        _ensure_training_run(db, logging_v2, run_id="test", started_at="2026-01-01",
                             backend="python", rule_set="normal", seed=0, config={},
                             metadata={}, resume=False)
        with pytest.raises(RuntimeError, match="tactical failed"):
            train._evaluate_promotion(
                trainer, checkpoint, tmp_path / "best.json", tmp_path / "latest.json",
                config_path="normal.yaml", tournament_db=tmp_path / "tournament.sqlite",
                run_id="test", seed=0, logging_db=db, logging_v2=logging_v2)
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 0
    assert trainer.promotion_eval_games == 0


def test_gate_rejects_crossed_boundary_without_running_evaluation(tmp_path):
    from types import SimpleNamespace
    from scripts import train

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    trainer = SimpleNamespace(n_iter=101, cfg={"games_per_iter": 50},
                              promotion_eval_games=0)
    with pytest.raises(ValueError, match="crossed"):
        train._evaluate_promotion(
            trainer, checkpoint, tmp_path / "best.json", tmp_path / "latest.json",
            config_path="normal.yaml", tournament_db=tmp_path / "tournament.sqlite",
            run_id="test", seed=0, logging_db=None, logging_v2=logging_v2)





def test_training_passes_backend_and_workers_to_promotion_gate(tmp_path, monkeypatch):
    from config import DEFAULT_CONFIG
    from scripts import train

    cfg = {**DEFAULT_CONFIG, "n_blocks": 1, "n_filters": 8, "n_iter": 0,
           "buffer_size": 32, "rule_set": "normal"}
    backend = {"requested_backend": "cpp", "mcts": "cpp", "patterns": "cpp"}
    calls = []
    monkeypatch.setattr(train, "load_config", lambda _: dict(cfg))
    monkeypatch.setattr("az.backend.configure_backends", lambda *args, **kwargs: backend)
    monkeypatch.setattr(train, "_evaluate_promotion",
                        lambda *args, **kwargs: calls.append(kwargs) or None)
    assert train.main([
        "--fresh", "--out-dir", str(tmp_path / "checkpoints"),
        "--db", str(tmp_path / "normal.sqlite"),
        "--tournament-db", str(tmp_path / "tournament.sqlite"),
        "--backend", "cpp", "--workers", "3", "--run-id", "gate-wiring",
    ]) == 0
    assert calls[0]["backends"] == backend
    assert calls[0]["workers"] == 3

def test_training_defaults_to_official_resume_and_fresh_is_explicit(tmp_path, monkeypatch):
    from scripts import train

    args = train.build_parser().parse_args([])
    assert args.out_dir == "checkpoints/normal"
    assert not args.fresh
    monkeypatch.chdir(tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        train.main(["--backend", "python", "--logging-level", "none"])
    assert not (tmp_path / "checkpoints" / "normal" / "latest.json").exists()


def test_fresh_and_resume_are_mutually_exclusive():
    from scripts.train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--fresh", "--resume"])

def test_bootstrap_options_are_not_accepted():
    from scripts.train import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--bootstrap", "old.pt"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--bootstrap-optimizer", "reset"])


def test_fresh_commits_initial_generation_and_run_metadata(tmp_path, monkeypatch):
    import json
    from az.model import load_checkpoint
    from az.trainer import resolve_generation
    from config import DEFAULT_CONFIG
    from scripts import train

    cfg = {**DEFAULT_CONFIG, "n_blocks": 1, "n_filters": 8, "n_iter": 0,
           "buffer_size": 32, "rule_set": "normal"}
    monkeypatch.setattr(train, "load_config", lambda _: dict(cfg))
    output = tmp_path / "new"
    database = tmp_path / "new.sqlite"
    assert train.main(["--fresh", "--out-dir", str(output), "--db", str(database),
                       "--tournament-db", str(tmp_path / "tournament.sqlite"),
                       "--backend", "python", "--run-id", "bootstrap-test"]) == 0
    checkpoint, replay = resolve_generation(output / "latest.json")
    _, meta = load_checkpoint(checkpoint, device="cpu")
    assert meta["global_step"] == 0 and meta["bootstrap"] is None
    from az.replay_buffer import ReplayBuffer
    assert len(ReplayBuffer.load_npz(replay)) == 0
    with logging_v2.connect(database) as db:
        run = db.execute("SELECT metadata_json FROM runs WHERE run_id='bootstrap-test'").fetchone()
        assert json.loads(run["metadata_json"])["bootstrap"] == meta["bootstrap"]
    assert not (tmp_path / "tournament.sqlite").exists()
    assert meta["run_id"] == "bootstrap-test"
    assert train.main(["--resume", "--out-dir", str(output), "--db", str(database),
                       "--tournament-db", str(tmp_path / "tournament.sqlite"),
                       "--backend", "python", "--iterations", "0"]) == 0
    with pytest.raises(ValueError, match="original --run-id"):
        train.main(["--resume", "--out-dir", str(output), "--db", str(database),
                    "--tournament-db", str(tmp_path / "tournament.sqlite"),
                    "--backend", "python", "--run-id", "wrong-run", "--iterations", "0"])
    with logging_v2.connect(database) as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 1
    assert train.main(["--resume", "--generation", checkpoint.parent.name,
                       "--out-dir", str(output), "--db", str(database),
                       "--tournament-db", str(tmp_path / "tournament.sqlite"),
                       "--backend", "python", "--run-id", "rollback-run", "--iterations", "0"]) == 0
    with logging_v2.connect(database) as db:
        assert db.execute("SELECT count(*) FROM runs").fetchone()[0] == 2
    with pytest.raises(ValueError, match="original --run-id"):
        train.main(["--resume", "--out-dir", str(output), "--db", str(database),
                    "--tournament-db", str(tmp_path / "tournament.sqlite"),
                    "--backend", "python", "--run-id", "rollback-run", "--iterations", "0"])

@pytest.mark.parametrize("crash_point", ["games", "pointer"])
def test_resume_after_games_commit_preserves_original_targets(tmp_path, monkeypatch, crash_point):
    import numpy as np
    from az.game_log import GameRecord
    from az.replay_buffer import ReplayBuffer
    from az.trainer import Trainer, resolve_generation
    from config import DEFAULT_CONFIG
    from scripts import train

    cfg = {**DEFAULT_CONFIG, "n_blocks": 1, "n_filters": 8, "n_iter": 1,
           "games_per_iter": 1, "buffer_size": 32, "batch_size": 1,
           "train_steps_per_iter": 1, "eval_fixed_batch_size": 1}
    monkeypatch.setattr(train, "load_config", lambda _: dict(cfg))
    calls = []

    def play(model, rules, config, n_games, seed, buffer, model_iter):
        calls.append(1)
        provenance = {**config["provenance"], "game_id": "a" * 64}
        policy = np.zeros(121, dtype=np.float32)
        policy[3] = 1
        buffer.add(np.zeros((4, 11, 11), dtype=np.uint8), policy, 1, "normal", provenance["game_id"])
        yield GameRecord(0, str(len(calls)), "normal", "az", "az", "black", True,
                         "five_in_row", 1, float(len(calls)), model_iter=model_iter,
                         config={"provenance": provenance})

    monkeypatch.setattr(train, "run_games", play)
    original = Trainer.save

    def crash(self, pointer):
        if self.n_iter == 1:
            raise RuntimeError("crash after games commit")
        return original(self, pointer)

    original_step = logging_v2.record_training_step
    if crash_point == "games":
        monkeypatch.setattr(Trainer, "save", crash)
    else:
        def crash_step(*args, **kwargs):
            raise RuntimeError("crash after pointer")
        monkeypatch.setattr(logging_v2, "record_training_step", crash_step)
    args = ["--out-dir", str(tmp_path / "out"), "--db", str(tmp_path / "db.sqlite"),
            "--tournament-db", str(tmp_path / "tour.sqlite"), "--backend", "python", "--run-id", "run"]
    with pytest.raises(RuntimeError, match="crash after"):
        train.main(args + ["--fresh"])
    if crash_point == "pointer":
        with pytest.raises(ValueError, match="original.*run"):
            train.main(args + ["--resume", "--run-id", "other", "--iterations", "1"])
        with logging_v2.connect(tmp_path / "db.sqlite") as db:
            assert db.execute("SELECT count(*) FROM runs WHERE run_id='other'").fetchone()[0] == 0
    with logging_v2.connect(tmp_path / "db.sqlite") as db:
        original_payload = db.execute("SELECT payload_json FROM games").fetchone()[0]
    monkeypatch.setattr(Trainer, "save", original)
    monkeypatch.setattr(logging_v2, "record_training_step", original_step)
    assert train.main(args + ["--resume", "--iterations", "1"]) == 0
    assert train.main(args + ["--resume", "--iterations", "1"]) == 0
    _, replay = resolve_generation(tmp_path / "out" / "latest.json")
    restored = ReplayBuffer.load_npz(replay)
    assert len(calls) == 1
    assert restored.game_ids[0] == "a" * 64
    assert restored.pi[0, 3] == 1 and restored.z[0] == 1
    with logging_v2.connect(tmp_path / "db.sqlite") as db:
        assert db.execute("SELECT payload_json FROM games").fetchone()[0] == original_payload
        assert db.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 1
    checkpoint, _ = resolve_generation(tmp_path / "out" / "latest.json")
    import torch
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"]
    assert meta["lineage"]["historical_games"] is None
    assert meta["lineage"]["generated_since_tracking"] == 1
    assert meta["lineage"]["identified_game_iteration_uses"] == 1
    assert meta["lineage"]["identified_game_ids_used_since_tracking"] == ["a" * 64]


def test_retry_interrupted_explicit_rollback_reuses_outbox(tmp_path, monkeypatch):
    import json
    import numpy as np
    from az.game_log import GameRecord
    from az.trainer import Trainer, resolve_generation
    from config import DEFAULT_CONFIG
    from scripts import train

    cfg = {**DEFAULT_CONFIG, "n_blocks": 1, "n_filters": 8, "n_iter": 1,
           "games_per_iter": 1, "buffer_size": 32, "batch_size": 1,
           "train_steps_per_iter": 1, "eval_fixed_batch_size": 1}
    monkeypatch.setattr(train, "load_config", lambda _: dict(cfg))
    calls = []

    def play(model, rules, config, n_games, seed, buffer, model_iter):
        run = config["provenance"]["run_id"]
        calls.append(run)
        provenance = {**config["provenance"], "game_id": ("a" if run == "source" else "b") * 64}
        policy = np.zeros(121, dtype=np.float32)
        policy[3] = 1
        buffer.add(np.zeros((4, 11, 11), dtype=np.uint8), policy, 1, "normal", provenance["game_id"])
        yield GameRecord(0, "time", "normal", "az", "az", "black", True,
                         "five_in_row", 1, 1, model_iter=model_iter, config={"provenance": provenance})

    monkeypatch.setattr(train, "run_games", play)
    args = ["--out-dir", str(tmp_path / "out"), "--db", str(tmp_path / "db.sqlite"),
            "--tournament-db", str(tmp_path / "tour.sqlite"), "--backend", "python"]
    assert train.main(args + ["--fresh", "--run-id", "source"]) == 0
    pointer = tmp_path / "out" / "latest.json"
    old, _ = resolve_generation(pointer)
    ancestor = json.loads((old.parent / "manifest.json").read_text())["previous"]
    rollback = args + ["--resume", "--generation", ancestor, "--run-id", "branch", "--iterations", "1"]
    original = Trainer.save

    def crash(self, path):
        if self.n_iter == 1:
            raise RuntimeError("crash after rollback games")
        return original(self, path)

    monkeypatch.setattr(Trainer, "save", crash)
    with pytest.raises(RuntimeError, match="crash after rollback games"):
        train.main(rollback)
    monkeypatch.setattr(Trainer, "save", original)
    assert train.main(rollback) == 0
    with pytest.raises(ValueError, match="already committed"):
        train.main(rollback)
    assert train.main(args + ["--resume", "--run-id", "branch", "--iterations", "1"]) == 0
    with pytest.raises(ValueError, match="same rollback source"):
        train.main(args + ["--resume", "--generation", old.parent.name, "--run-id", "branch", "--iterations", "1"])
    assert calls == ["source", "branch"]
    assert resolve_generation(pointer, old.parent.name)[0] == old
    with logging_v2.connect(tmp_path / "db.sqlite") as db:
        assert db.execute("SELECT count(*) FROM games WHERE run_id='branch'").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM training_steps WHERE run_id='branch'").fetchone()[0] == 1


def test_recovery_rejects_source_collision_and_tampering(tmp_path):
    from az import training_recovery as recovery
    from az.game_log import GameRecord
    from az.replay_buffer import ReplayBuffer

    identity = recovery._identity("run", 0, "source")
    provenance = {**identity, "game_id": "a" * 64}
    record = GameRecord(0, "time", "normal", "az", "az", "black", True,
                        "five_in_row", 1, 1, config={"provenance": provenance})
    directory = recovery.prepare(tmp_path, identity, ReplayBuffer(8), [record])
    with pytest.raises(ValueError, match="source checkpoint collision"):
        recovery.reject_other_source(tmp_path, recovery._identity("run", 0, "other"))
    recovery.reject_other_source(tmp_path, recovery._identity("branch", 0, "other"))
    with (directory / "records.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="hash mismatch"):
        recovery.load(tmp_path, identity)


def test_recovery_rejects_existing_game_with_different_owner(tmp_path):
    from az.game_log import GameRecord
    from scripts.train import _record_games

    database = tmp_path / "games.sqlite"
    logging_v2.initialize(database, "selfplay")
    record = GameRecord(0, "time", "normal", "az", "az", "black", True,
                        "five_in_row", 1, 1,
                        config={"provenance": {"game_id": "a" * 64}})
    with logging_v2.connect(database) as db:
        for run in ("first", "second"):
            logging_v2.record_selfplay_run(db, run_id=run, started_at="time", backend="python",
                                           rule_set="normal", seed=0, config={})
        _record_games(db, logging_v2, "first", 0, 1, [record])
        before = tuple(db.execute("SELECT * FROM games").fetchone())
        with pytest.raises(ValueError, match="game identity collision"):
            _record_games(db, logging_v2, "second", 0, 1, [record])
        assert tuple(db.execute("SELECT * FROM games").fetchone()) == before


def test_process_exit_after_sqlite_commit_resumes_without_regenerating(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path
    from az.replay_buffer import ReplayBuffer
    from az.trainer import resolve_generation

    script = r"""
import os, sys, numpy as np
from az.game_log import GameRecord
from az.trainer import Trainer
from config import DEFAULT_CONFIG
from scripts import train
root, mode = sys.argv[1:]
cfg = {**DEFAULT_CONFIG, "n_blocks": 1, "n_filters": 8, "n_iter": 1,
       "games_per_iter": 1, "buffer_size": 32, "batch_size": 1,
       "train_steps_per_iter": 1, "eval_fixed_batch_size": 1}
train.load_config = lambda _: dict(cfg)
def play(model, rules, config, n_games, seed, buffer, model_iter):
    if mode != "crash":
        raise AssertionError("resume regenerated self-play")
    provenance = {**config["provenance"], "game_id": "b" * 64}
    policy = np.zeros(121, dtype=np.float32); policy[7] = 1
    buffer.add(np.zeros((4, 11, 11), dtype=np.uint8), policy, 1, "normal", provenance["game_id"])
    yield GameRecord(0, "original", "normal", "az", "az", "black", True,
                     "five_in_row", 1, 1, model_iter=model_iter, config={"provenance": provenance})
train.run_games = play
save = Trainer.save
def crash(self, pointer):
    if self.n_iter == 1:
        os._exit(73)
    return save(self, pointer)
if mode == "crash":
    Trainer.save = crash
args = ["--out-dir", root + "/out", "--db", root + "/db.sqlite",
        "--tournament-db", root + "/tour.sqlite", "--backend", "python", "--run-id", "run"]
train.main(args + (["--resume", "--iterations", "1"] if mode == "resume" else ["--fresh"]))
"""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1"}
    cwd = Path(__file__).resolve().parents[1]
    crashed = subprocess.run([sys.executable, "-c", script, str(tmp_path), "crash"],
                             cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    assert crashed.returncode == 73, crashed.stderr
    with logging_v2.connect(tmp_path / "db.sqlite") as db:
        payload = db.execute("SELECT payload_json FROM games").fetchone()[0]
    for _ in range(2):
        resumed = subprocess.run([sys.executable, "-c", script, str(tmp_path), "resume"],
                                 cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
        assert resumed.returncode == 0, resumed.stderr
    _, replay_path = resolve_generation(tmp_path / "out" / "latest.json")
    replay = ReplayBuffer.load_npz(replay_path)
    assert len(replay) == 1 and replay.game_ids[0] == "b" * 64
    assert replay.pi[0, 7] == 1 and replay.z[0] == 1
    with logging_v2.connect(tmp_path / "db.sqlite") as db:
        assert db.execute("SELECT payload_json FROM games").fetchone()[0] == payload
        assert db.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 1


def test_recovery_keeps_samples_overwritten_by_replay_ring(tmp_path):
    import numpy as np
    from az import training_recovery as recovery
    from az.game_log import GameRecord
    from az.replay_buffer import ReplayBuffer

    identity = recovery._identity("run", 0, "source")
    capture = recovery.Capture(ReplayBuffer(1), 2)
    records = []
    policy = np.zeros(121, dtype=np.float32)
    policy[3] = 1
    for index in range(2):
        game_id = str(index) * 64
        capture.add(np.full((4, 11, 11), index, dtype=np.uint8), policy, index, "normal", game_id)
        records.append(GameRecord(index, "time", "normal", "az", "az", "black", True,
                                  "five_in_row", 1, 1,
                                  config={"provenance": {**identity, "game_id": game_id}}))
    directory = recovery.prepare(tmp_path, identity, capture.buffer, records, capture.samples)
    samples = ReplayBuffer.load_npz(directory / "samples.npz")
    assert len(capture.buffer) == 1 and len(samples) == 2
    assert list(samples.z[:2]) == [0, 1]
    assert list(samples.game_ids[:2]) == ["0" * 64, "1" * 64]
