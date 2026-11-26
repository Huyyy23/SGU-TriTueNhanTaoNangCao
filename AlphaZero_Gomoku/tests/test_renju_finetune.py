import json
import numpy as np
import pytest
import torch

from az.model import AZNet, save_checkpoint
from az.replay_buffer import ReplayBuffer
from az.selfplay import run_games
from az.trainer import Trainer
from config import load_config
from gomoku.board import Board
from gomoku.rules import get_rules


def test_renju_config_has_full_budget_and_finetune_settings():
    cfg = load_config("configs/renju.yaml")
    assert cfg["n_iter"] * cfg["games_per_iter"] == 5000
    assert cfg["learning_rate"] == pytest.approx(1e-3)
    assert (cfg["n_blocks"], cfg["n_filters"], cfg["n_simulations"]) == (6, 64, 200)
    assert cfg["buffer_size"] >= cfg["batch_size"] and cfg["checkpoint_buffer"]
    assert cfg["selfplay_opening_plies"] == 4
    assert cfg["dirichlet_eps"] == pytest.approx(0.35)
    assert 0.0 < cfg["selfplay_random_opponent_rate"] < 1.0
    assert cfg["outcome_guard_min_games"] == 500
    assert cfg["outcome_guard_max_side_win_rate"] == pytest.approx(0.98)


def _official_source(tmp_path, iteration=1500, rule_set="normal"):
    cfg = {**load_config("configs/renju.yaml"), "rule_set": "normal", "n_blocks": 1,
           "n_filters": 8, "n_iter": 1, "train_steps_per_iter": 1}
    trainer = Trainer(AZNet(1, 8), cfg, buffer=ReplayBuffer(16))
    trainer.n_iter = iteration
    source = tmp_path / "normal" / "latest.json"
    trainer.save(source)
    checkpoint, _ = __import__("az.trainer", fromlist=["resolve_generation"]).resolve_generation(source)
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    data["meta"]["rule_set"] = rule_set
    data["meta"]["n_iter"] = iteration
    torch.save(data, checkpoint)
    manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    import hashlib
    manifest["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (checkpoint.parent / "manifest.json").write_text(json.dumps(manifest))
    return source, checkpoint


@pytest.mark.parametrize(("iteration", "rule_set"), [(1499, "normal"), (1500, "renju")])
def test_main_rejects_non_official_source(tmp_path, monkeypatch, iteration, rule_set):
    import az.renju_finetune as lifecycle

    source, _ = _official_source(tmp_path, iteration=iteration, rule_set=rule_set)
    with pytest.raises(SystemExit, match="Normal checkpoint at iteration 1500"):
        lifecycle.main([
            "--source", str(source), "--out-dir", str(tmp_path / "renju"),
            "--db", str(tmp_path / "renju.sqlite"),
        ])


def test_renju_paths_cannot_overlap_normal_artifacts(tmp_path):
    from pathlib import Path

    import az.renju_finetune as lifecycle

    with pytest.raises(ValueError, match="must not overlap Normal"):
        lifecycle._validate_paths(Path("checkpoints/normal"), tmp_path / "renju.sqlite")
    with pytest.raises(ValueError, match="must not overlap Normal"):
        lifecycle._validate_paths(tmp_path / "renju", Path("data/normal.sqlite"))


def test_from_pretrained_starts_empty_renju_buffer(tmp_path):
    source, checkpoint = _official_source(tmp_path)
    cfg = {**load_config("configs/renju.yaml"), "n_blocks": 1, "n_filters": 8}
    buffer = ReplayBuffer(32)
    original = checkpoint.read_bytes()
    trainer = Trainer.from_pretrained(checkpoint, cfg, buffer=buffer, seed=19)
    assert trainer.cfg["rule_set"] == "renju"
    assert trainer.n_iter == trainer.global_step == 0
    assert len(buffer) == 0
    assert checkpoint.read_bytes() == original


def test_cpp_renju_path_excludes_black_forbidden_targets_and_moves():
    from az.mcts import NativeMCTS

    board = Board()
    board.grid[[5 * 11 + c for c in (4, 6)] + [r * 11 + 5 for r in (4, 6)]] = 1
    rules = get_rules("renju")
    forbidden = set(rules.forbidden_moves(board))
    native = NativeMCTS(rules, n_simulations=2, root_noise=False, rng=np.random.default_rng(1))
    native.reset(board)
    request = native.request()
    assert request is not None and not request.legal_mask[list(forbidden)].any()
    native.receive(np.ones(121, dtype=np.float32), 0.0)
    assert not forbidden.intersection(native.visit_counts().nonzero()[0])


def test_selfplay_run_receives_renju_rules_and_keeps_forbidden_targets_masked():
    class Model:
        def predict(self, features):
            return np.ones((len(features), 121), dtype=np.float32), np.zeros(len(features), dtype=np.float32)

    cfg = {"rule_set": "renju", "parallel_games": 1, "n_simulations": 2, "c_puct": 1.5,
           "dirichlet_alpha": 0.15, "dirichlet_eps": 0.25, "pi_target_temp": 1.0,
           "temp_moves": 0, "mcts_backend": "python", "patterns_backend": "python",
           "mcts_logging_level": "STANDARD"}
    rules = get_rules("renju")
    buffer = ReplayBuffer(50000)
    records = list(run_games(Model(), rules, cfg, 1, seed=3, buffer=buffer))
    assert records and records[0].rule_set == "renju"
    assert all(record.moves_count > 0 for record in records)
    for index in range(len(buffer)):
        forbidden = buffer.features[index, 2].reshape(-1).astype(bool)
        occupied = buffer.features[index, :2].any(axis=0).reshape(-1)
        assert not buffer.pi[index, forbidden].any()
        assert not buffer.pi[index, occupied].any()
    board = Board()
    for move in records[0].moves:
        action = move.row * 11 + move.col
        assert action in rules.legal_moves(board)
        board.play(move.row, move.col)


def test_main_recovers_committed_generation_without_duplicate_games(tmp_path, monkeypatch):
    import az.renju_finetune as lifecycle
    from az.logging_v2 import connect
    from az.trainer import resolve_generation

    cfg = {
        **load_config("configs/renju.yaml"),
        "n_iter": 1,
        "games_per_iter": 1,
        "train_steps_per_iter": 1,
        "batch_size": 1,
        "eval_fixed_batch_size": 1,
        "buffer_size": 32,
        "n_blocks": 1,
        "n_filters": 8,
        "n_simulations": 2,
        "parallel_games": 1,
    }
    monkeypatch.setattr(lifecycle, "load_config", lambda _path: dict(cfg))
    backends = {"requested_backend": "python", "mcts": "python", "patterns": "python"}
    monkeypatch.setattr(lifecycle, "configure_backends", lambda *args, **kwargs: backends)
    source, source_checkpoint = _official_source(tmp_path)
    source_bytes = source_checkpoint.read_bytes()
    out_dir = tmp_path / "renju"
    database = tmp_path / "renju.sqlite"
    argv = [
        "--source", str(source), "--out-dir", str(out_dir), "--db", str(database),
        "--backend", "python", "--iterations", "1", "--games", "1",
    ]

    record_training_step = lifecycle.record_training_step
    monkeypatch.setattr(
        lifecycle, "record_training_step",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("simulated interruption")),
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        lifecycle.main(argv)

    monkeypatch.setattr(lifecycle, "record_training_step", record_training_step)
    assert lifecycle.main([*argv, "--resume"]) == 0
    checkpoint, replay_path = resolve_generation(out_dir / "latest.json")
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"]
    replay = ReplayBuffer.load_npz(replay_path)
    assert meta["n_iter"] == 1 and meta["rule_set"] == "renju"
    assert meta["bootstrap"]["source"]["checkpoint_sha256"]
    assert len(replay) > 0
    assert source_checkpoint.read_bytes() == source_bytes
    with connect(database) as db:
        assert db.execute("SELECT count(*) FROM games").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 1


def test_main_respects_whole_batch_game_cap(tmp_path, monkeypatch):
    import az.renju_finetune as lifecycle
    from az.logging_v2 import connect
    from az.trainer import resolve_generation

    cfg = {
        **load_config("configs/renju.yaml"),
        "n_iter": 5,
        "games_per_iter": 2,
        "train_steps_per_iter": 1,
        "batch_size": 1,
        "eval_fixed_batch_size": 1,
        "buffer_size": 32,
        "n_blocks": 1,
        "n_filters": 8,
        "n_simulations": 2,
        "parallel_games": 1,
    }
    monkeypatch.setattr(lifecycle, "load_config", lambda _path: dict(cfg))
    backends = {"requested_backend": "python", "mcts": "python", "patterns": "python"}
    monkeypatch.setattr(lifecycle, "configure_backends", lambda *args, **kwargs: backends)
    source, source_checkpoint = _official_source(tmp_path)
    source_bytes = source_checkpoint.read_bytes()
    out_dir = tmp_path / "renju"
    database = tmp_path / "renju.sqlite"
    argv = [
        "--source", str(source), "--out-dir", str(out_dir), "--db", str(database),
        "--backend", "python", "--iterations", "5", "--games", "2",
        "--max-new-games", "5",
    ]

    assert lifecycle.main(argv) == 0
    checkpoint, replay_path = resolve_generation(out_dir / "latest.json")
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"]
    assert meta["n_iter"] == 2
    assert meta["config"]["n_iter"] == 5
    with connect(database) as db:
        assert db.execute("SELECT count(*) FROM games").fetchone()[0] == 4
        assert db.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 2
        rows = db.execute("SELECT metrics_json FROM training_steps").fetchall()
    for (metrics_json,) in rows:
        metrics = json.loads(metrics_json)
        assert metrics["start_iteration"] == 0
        assert metrics["game_limit"] == 5
        assert metrics["target_iteration"] == 2
    assert source_checkpoint.read_bytes() == source_bytes

    resume_argv = [
        "--source", str(source), "--out-dir", str(out_dir), "--db", str(database),
        "--backend", "python", "--iterations", "5", "--games", "2",
        "--max-new-games", "2", "--resume",
    ]
    assert lifecycle.main(resume_argv) == 0
    checkpoint, _ = resolve_generation(out_dir / "latest.json")
    meta = torch.load(checkpoint, map_location="cpu", weights_only=False)["meta"]
    assert meta["n_iter"] == 3
    assert meta["config"]["n_iter"] == 5
    with connect(database) as db:
        assert db.execute("SELECT count(*) FROM games").fetchone()[0] == 6
        assert db.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 3
        last = db.execute("SELECT metrics_json FROM training_steps ORDER BY global_step DESC LIMIT 1").fetchone()[0]
    last_metrics = json.loads(last)
    assert last_metrics["start_iteration"] == 2
    assert last_metrics["game_limit"] == 2
    assert last_metrics["target_iteration"] == 3


def test_main_stops_on_one_sided_outcomes(tmp_path, monkeypatch):
    import az.renju_finetune as lifecycle
    from az.logging_v2 import connect

    cfg = {
        **load_config("configs/renju.yaml"),
        "n_iter": 5,
        "games_per_iter": 1,
        "train_steps_per_iter": 1,
        "batch_size": 1,
        "eval_fixed_batch_size": 1,
        "buffer_size": 32,
        "n_blocks": 1,
        "n_filters": 8,
        "n_simulations": 2,
        "parallel_games": 1,
        "outcome_guard_min_games": 1,
        "outcome_guard_max_side_win_rate": 0.51,
    }
    monkeypatch.setattr(lifecycle, "load_config", lambda _path: dict(cfg))
    backends = {"requested_backend": "python", "mcts": "python", "patterns": "python"}
    monkeypatch.setattr(lifecycle, "configure_backends", lambda *args, **kwargs: backends)
    source, _ = _official_source(tmp_path)
    out_dir = tmp_path / "renju"
    database = tmp_path / "renju.sqlite"
    argv = [
        "--source", str(source), "--out-dir", str(out_dir), "--db", str(database),
        "--backend", "python", "--iterations", "5", "--games", "1",
    ]

    assert lifecycle.main(argv) == 3
    with connect(database) as db:
        assert db.execute("SELECT count(*) FROM games").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 1
        event = db.execute(
            "SELECT payload_json FROM events WHERE event='outcome_guard_stop'"
        ).fetchone()
    assert event is not None
    assert "degenerate self-play outcome" in json.loads(event[0])["reason"]

    # Resume must stop before spending another game on the same distribution.
    assert lifecycle.main([*argv, "--resume"]) == 3
    with connect(database) as db:
        assert db.execute("SELECT count(*) FROM games").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 1

