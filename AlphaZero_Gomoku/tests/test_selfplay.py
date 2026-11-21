import json

from az.logging_v2 import decode_mcts_root

import numpy as np
import pytest
from concurrent.futures import ProcessPoolExecutor

from az.model import AZNet
from az.replay_buffer import RULE_SET_IDS, ReplayBuffer
from az.selfplay import run_games, run_games_parallel, split_games
from config import DEFAULT_CONFIG
from az.logging_v2 import connect, initialize, record_selfplay_game, record_selfplay_run
from gomoku.board import Board, SIZE
from gomoku.features import board_to_planes
from gomoku.rules import get_rules
from gomoku.features import N_ACTIONS
from gomoku.rules.normal import NormalRules


class UniformEvaluator:
    """Fake evaluator: uniform policy over 121 cells, value 0. No torch, no training."""

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch = features.shape[0]
        return (
            np.full((batch, N_ACTIONS), 1.0 / N_ACTIONS, dtype=np.float32),
            np.zeros(batch, dtype=np.float32),
        )


def small_cfg(**overrides) -> dict:
    return {**DEFAULT_CONFIG, "n_simulations": 8, "temp_moves": 4, "rule_set": "normal", **overrides}

def test_backend_is_resolved_once_for_reused_selfplay_config(monkeypatch):
    import az.selfplay as selfplay_module

    calls = 0
    original = selfplay_module.configure_backends

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(selfplay_module, "configure_backends", counted)
    cfg = small_cfg(backend="python", mcts_backend="python", patterns_backend="python")
    for seed in (1, 2):
        list(run_games(UniformEvaluator(), NormalRules(), cfg, 1, seed=seed))

    assert calls == 1


def test_one_game_sample_count_matches_moves():
    buffer = ReplayBuffer(capacity=1000)
    records = list(run_games(UniformEvaluator(), NormalRules(), small_cfg(), 1, seed=3, buffer=buffer))
    record = records[0]

    assert record.validate() is None
    assert record.moves_count == len(buffer) > 0
    assert record.nn_calls > 0
    assert record.mean_batch == 1.0
    assert set(buffer.rule_set[: len(buffer)].tolist()) == {RULE_SET_IDS["normal"]}


def test_z_alternates_sign_with_player_to_move():
    buffer = ReplayBuffer(capacity=1000)
    record = list(run_games(UniformEvaluator(), NormalRules(), small_cfg(), 1, seed=3, buffer=buffer))[0]

    z = buffer.z[: len(buffer)]
    if record.winner == "draw":
        assert np.all(z == 0)
    else:
        assert np.all(z[:-1] * z[1:] == -1)
        first_player_z = 1.0 if record.winner == "black" else -1.0
        assert z[0] == first_player_z


def test_game_is_decisive_and_logged_with_correct_end_reason():
    buffer = ReplayBuffer(capacity=1000)
    record = list(run_games(UniformEvaluator(), NormalRules(), small_cfg(), 1, seed=3, buffer=buffer))[0]

    assert record.winner in {"black", "white"}
    assert record.end_reason == "five_in_row"
    assert record.first_player_won == (record.winner == "black")


def test_same_seed_gives_same_game():
    cfg = small_cfg()
    first = list(run_games(UniformEvaluator(), NormalRules(), cfg, 1, seed=11))[0]
    second = list(run_games(UniformEvaluator(), NormalRules(), cfg, 1, seed=11))[0]
    assert (first.winner, first.moves_count, first.nn_calls) == (second.winner, second.moves_count, second.nn_calls)


def test_standard_root_logging_preserves_moves_and_captures_search_data(tmp_path):
    cfg = small_cfg(n_simulations=3)
    baseline_buffer = ReplayBuffer(1000)
    logged_buffer = ReplayBuffer(1000)
    baseline = next(run_games(UniformEvaluator(), NormalRules(), cfg, 1, seed=19, buffer=baseline_buffer))
    assert baseline.moves is None and baseline.root_data is None
    logged_cfg = {**cfg, "mcts_logging_level": "STANDARD"}
    logged = next(run_games(UniformEvaluator(), NormalRules(), logged_cfg, 1, seed=19, buffer=logged_buffer))
    np.testing.assert_array_equal(baseline_buffer.features, logged_buffer.features)
    np.testing.assert_array_equal(baseline_buffer.pi, logged_buffer.pi)
    np.testing.assert_array_equal(baseline_buffer.z, logged_buffer.z)
    assert logged.moves is not None and len(logged.moves) == logged.moves_count
    assert logged.root_data is not None and len(logged.root_data) == logged.moves_count
    root = decode_mcts_root(logged.root_data[0]["encoding"], logged.root_data[0]["data"])
    assert root["visit_counts"].shape == (121,)
    assert root["root_policy"].shape == (121,)
    assert root["target_policy"].shape == (121,)
    assert np.isclose(root["root_policy"].sum(), 1.0)
    assert logged.moves[0].temperature == 1.0
    assert root["legal_mask"].any()
    assert root["forbidden_mask"].shape == (121,)
    database = tmp_path / "selfplay.sqlite"
    initialize(database, "selfplay")
    with connect(database) as connection:
        record_selfplay_run(
            connection, run_id="run", started_at="start", backend="python",
            rule_set="normal", seed=19, config=logged.config,
        )
        record_selfplay_game(connection, "run", {
            "game_id": logged.game_id, "rule_set": logged.rule_set, "winner": logged.winner,
            "end_reason": logged.end_reason, "moves_count": logged.moves_count,
            "duration_sec": logged.duration_sec, "config": logged.config,
            "moves": [vars(move) for move in logged.moves],
            "root_data": logged.root_data,
        })
        assert connection.execute("SELECT count(*) FROM mcts_roots").fetchone()[0] == logged.moves_count


def test_n_games_yields_n_records_and_ids_are_sequential():
    records = list(run_games(UniformEvaluator(), NormalRules(), small_cfg(n_simulations=3), 2, seed=5))
    assert [r.game_id for r in records] == [0, 1]


def test_rule_set_mismatch_raises():
    cfg = small_cfg(rule_set="renju")
    with pytest.raises(ValueError):
        list(run_games(UniformEvaluator(), NormalRules(), cfg, 1, seed=0))


def test_twenty_moves_under_five_seconds():
    cfg = small_cfg(n_simulations=200)
    record = list(run_games(UniformEvaluator(), NormalRules(), cfg, 1, seed=0))[0]
    seconds_per_20_moves = record.duration_sec / record.moves_count * 20
    assert seconds_per_20_moves < 5.0, (
        f"{seconds_per_20_moves:.2f}s per 20 moves "
        f"({record.duration_sec:.2f}s for {record.moves_count} moves, {cfg['n_simulations']} sims)"
    )


def test_parallel_games_batches_leaves_and_keeps_search_budget_per_move():
    single = list(run_games(UniformEvaluator(), NormalRules(), small_cfg(parallel_games=1), 3, seed=3))
    batched = list(run_games(UniformEvaluator(), NormalRules(), small_cfg(parallel_games=4), 3, seed=3))

    assert all(record.mean_batch == 1.0 for record in single)
    assert all(record.mean_batch > 1.0 for record in batched)

    ratio_single = sum(r.nn_calls / r.moves_count for r in single) / len(single)
    ratio_batched = sum(r.nn_calls / r.moves_count for r in batched) / len(batched)
    assert 6.5 <= ratio_batched <= 8.0
    assert abs(ratio_batched - ratio_single) < 0.1


def test_parallel_games_plays_exactly_n_games():
    cfg = small_cfg(parallel_games=3, n_simulations=3)
    records = list(run_games(UniformEvaluator(), NormalRules(), cfg, 7, seed=5))
    # Records finish out of order; ids are assigned when a game starts.
    assert sorted(r.game_id for r in records) == list(range(7))


def test_records_carry_model_iter_and_config():
    cfg = small_cfg(n_simulations=3)
    record = list(run_games(UniformEvaluator(), NormalRules(), cfg, 1, seed=1, model_iter=4))[0]
    assert record.model_iter == 4
    assert record.config["n_simulations"] == 3


def test_split_games_is_even_and_covers_every_game():
    assert split_games(50, 4) == [13, 13, 12, 12]
    assert sum(split_games(50, 4)) == 50
    assert split_games(3, 8) == [1, 1, 1]  # never more chunks than games
    assert split_games(0, 4) == []


def test_parallel_selfplay_stores_every_move_once():
    cfg = small_cfg(n_simulations=2, n_blocks=1, n_filters=8)
    buffer = ReplayBuffer(capacity=1000)
    model = AZNet(n_blocks=1, n_filters=8)

    with ProcessPoolExecutor(max_workers=2) as pool:
        records = run_games_parallel(pool, model, NormalRules(), cfg, 3, seed=4, buffer=buffer, workers=2)

    assert len(records) == 3
    assert sorted(r.game_id for r in records) == [0, 1, 2]
    assert len(buffer) == sum(r.moves_count for r in records) > 0

    # A lost sample and a duplicated sample cancel out on the count, so key on content too.
    keys = {
        buffer.features[i].tobytes() + buffer.pi[i].tobytes() + buffer.z[i : i + 1].tobytes()
        for i in range(len(buffer))
    }
    assert len(keys) == len(buffer)


def test_parallel_worker_keeps_configured_active_game_concurrency():
    cfg = small_cfg(n_simulations=2, n_blocks=1, n_filters=8, parallel_games=1)
    model = AZNet(n_blocks=1, n_filters=8)

    with ProcessPoolExecutor(max_workers=2) as pool:
        records = run_games_parallel(pool, model, NormalRules(), cfg, 4, seed=4, workers=2)

    assert len(records) == 4
    assert all(record.mean_batch == 1.0 for record in records)


def test_parallel_provenance_links_each_sample_to_loaded_worker():
    from az.selfplay import _tensor_digest

    cfg = small_cfg(n_simulations=2, n_blocks=1, n_filters=8,
                    provenance={"run_id": "test-run", "source_checkpoint_sha256": "a" * 64})
    model = AZNet(n_blocks=1, n_filters=8)
    buffer = ReplayBuffer(capacity=1000)
    with ProcessPoolExecutor(max_workers=2) as pool:
        records = run_games_parallel(pool, model, NormalRules(), cfg, 3, seed=42,
                                     buffer=buffer, model_iter=7, workers=2)
    provenance = [r.config["provenance"] for r in records]
    assert len({p["game_id"] for p in provenance}) == 3
    assert len({p["game_seed"] for p in provenance}) == 3
    assert all(p["worker_model_sha256"] == _tensor_digest(model.state_dict()) for p in provenance)
    assert all(p["source_checkpoint_sha256"] == "a" * 64 and p["iteration"] == 7
               and p["run_id"] == "test-run" and p["ruleset"] == "normal" for p in provenance)
    for record in records:
        assert (buffer.game_ids[:len(buffer)] == record.config["provenance"]["game_id"]).sum() == record.moves_count


def test_worker_rejects_tensor_digest_mismatch():
    from az.selfplay import _play_chunk

    model = AZNet(n_blocks=1, n_filters=8)
    cfg = small_cfg(n_simulations=2, n_blocks=1, n_filters=8,
                    _expected_model_sha256="0" * 64, patterns_backend="python")
    with pytest.raises(ValueError, match="tensor digest"):
        _play_chunk((model.state_dict(), "normal", cfg, 1, 42, 0))


def test_game_seeds_are_reproducible_and_independent_of_batching():
    cfg = small_cfg(n_simulations=2, provenance={"run_id": "repro"})
    first = list(run_games(UniformEvaluator(), NormalRules(), {**cfg, "parallel_games": 1}, 2, seed=9))
    second = list(run_games(UniformEvaluator(), NormalRules(), {**cfg, "parallel_games": 2}, 2, seed=9))
    def identities(records):
        return {r.game_id: (r.config["provenance"]["game_id"], r.config["provenance"]["game_seed"])
                for r in records}
    assert identities(first) == identities(second)
    assert len({item[1] for item in identities(first).values()}) == 2


@pytest.mark.parametrize("opening_plies", [0, 2])
@pytest.mark.parametrize("rule_set", ["normal", "renju"])
def test_opening_telemetry_replays_only_mcts_samples(tmp_path, opening_plies, rule_set):
    rules = get_rules(rule_set)
    cfg = small_cfg(rule_set=rule_set, n_simulations=3, temp_moves=3,
                    selfplay_opening_plies=opening_plies, mcts_logging_level="STANDARD",
                    provenance={"run_id": "opening-test", "source_checkpoint_sha256": "a" * 64})
    samples = []
    record = next(run_games(UniformEvaluator(), rules, cfg, 1, seed=37, model_iter=4, samples_out=samples))
    provenance = record.config["provenance"]
    assert len(provenance["opening_actions"]) == opening_plies
    assert provenance["source_checkpoint_sha256"] == "a" * 64
    assert provenance["iteration"] == 4 and provenance["game_id"]
    assert len(samples) == record.moves_count == len(record.moves) == len(record.root_data)
    assert record.moves[0].move_index == opening_plies
    assert record.moves[0].temperature == 1.0
    assert record.moves[1].temperature == (0.0 if opening_plies == 2 else 1.0)

    database = tmp_path / "opening.sqlite"
    initialize(database, "selfplay")
    with connect(database) as connection:
        record_selfplay_run(connection, run_id="opening-test", started_at="start", backend="python",
                            rule_set=rule_set, seed=37, config=record.config)
        record_selfplay_game(connection, "opening-test", {
            "game_id": record.game_id, "rule_set": record.rule_set, "winner": record.winner,
            "end_reason": record.end_reason, "moves_count": record.moves_count,
            "duration_sec": record.duration_sec, "config": record.config,
            "moves": [vars(move) for move in record.moves], "root_data": record.root_data,
        })
        payload = json.loads(connection.execute("SELECT payload_json FROM games").fetchone()[0])
        rows = connection.execute("SELECT move_index, player, action FROM moves ORDER BY move_index").fetchall()
        assert connection.execute("SELECT count(*) FROM mcts_roots").fetchone()[0] == len(samples)

    saved = payload["config"]["provenance"]
    assert saved == provenance
    board = Board()
    rng = np.random.default_rng(saved["game_seed"])
    for action in saved["opening_actions"]:
        legal = rules.legal_moves(board)
        assert action == int(rng.choice(legal))
        assert action in legal
        board.play(*divmod(action, SIZE))
        assert not rules.is_terminal(board) and not rules.is_win(board)
    for (index, player, action), (planes, pi, z) in zip(rows, samples):
        assert index == len(board.history) and player == board.current_player
        legal = rules.legal_moves(board)
        assert action in legal
        np.testing.assert_array_equal(planes, board_to_planes(board, rules, player))
        assert np.isclose(pi.sum(), 1.0) and np.all(pi >= 0)
        assert not np.any(pi[np.setdiff1d(np.arange(N_ACTIONS), legal)])
        expected_z = 0 if record.winner == "draw" else (1 if (record.winner == "black") == (player == 1) else -1)
        assert z == expected_z
        board.play(*divmod(action, SIZE))
    assert rules.is_terminal(board)
    assert len(board.history) == record.moves_count + opening_plies


@pytest.mark.parametrize("value", [-1, 1, 3, 2.0, True, "2", None])
def test_selfplay_rejects_invalid_opening_config(value):
    with pytest.raises(ValueError, match="selfplay_opening_plies"):
        next(run_games(UniformEvaluator(), NormalRules(), small_cfg(selfplay_opening_plies=value), 1))


def test_openings_are_deterministic_across_batching_and_processes():
    cfg = small_cfg(n_simulations=2, n_blocks=1, n_filters=8, selfplay_opening_plies=2,
                    provenance={"run_id": "opening-repro"}, mcts_logging_level="STANDARD")
    model = AZNet(n_blocks=1, n_filters=8)
    first = list(run_games(model, NormalRules(), {**cfg, "parallel_games": 1}, 2, seed=9))
    second = list(run_games(model, NormalRules(), {**cfg, "parallel_games": 2}, 2, seed=9))
    buffer = ReplayBuffer(capacity=1000)
    with ProcessPoolExecutor(max_workers=2) as pool:
        parallel = run_games_parallel(pool, model, NormalRules(), cfg, 2, seed=9, buffer=buffer, workers=2)

    def openings(records):
        return {r.game_id: (r.config["provenance"]["game_seed"],
                            r.config["provenance"]["opening_actions"]) for r in records}

    assert openings(first) == openings(second) == openings(parallel)
    assert "opening_actions" not in cfg["provenance"]
    assert len({tuple(item[1]) for item in openings(parallel).values()}) == 2
    assert len(buffer) == sum(r.moves_count for r in parallel)
    for record in parallel:
        indices = np.flatnonzero(buffer.game_ids[:len(buffer)] == record.config["provenance"]["game_id"])
        assert len(indices) == record.moves_count
        assert all(buffer.features[i, :2].sum() >= 2 for i in indices)
def cpp_available() -> bool:
    try:
        import gomoku_mcts_native  # noqa: F401
        return True
    except ImportError:
        return False


cpp_only = pytest.mark.skipif(not cpp_available(), reason="gomoku_mcts_native extension not available")


@cpp_only
def test_cpp_selfplay_normal_runs_and_records_components():
    from gomoku.rules.normal import NormalRules

    cfg = small_cfg(n_simulations=4, mcts_backend="cpp", patterns_backend="python",
                    components_used={"mcts": "cpp", "patterns": "not_used"})
    records = list(run_games(UniformEvaluator(), NormalRules(), cfg, 2, seed=5))
    assert len(records) == 2
    for record in records:
        assert record.moves_count > 0
        assert record.nn_calls > 0
        assert record.config["components_used"]["mcts"] == "cpp"
        assert set(record.config["provenance"].keys()) >= {"run_id", "game_seed"}


@cpp_only
def test_cpp_selfplay_normal_runs_with_valid_samples():
    from gomoku.rules.normal import NormalRules

    base = small_cfg(n_simulations=4, temp_moves=0, parallel_games=1,
                     components_used={"mcts": "cpp", "patterns": "not_used"})
    samples = []
    records = list(run_games(UniformEvaluator(), NormalRules(), base, 1, seed=11,
                             samples_out=samples))
    assert len(records) == 1
    assert records[0].moves_count > 0
    assert records[0].nn_calls > 0
    assert records[0].config["components_used"]["mcts"] == "cpp"
    assert len(samples) == records[0].moves_count
    for _, pi, z in samples:
        assert np.isclose(pi.sum(), 1.0) and z in (-1.0, 0.0, 1.0)

@cpp_only
def test_cpp_selfplay_renju_forbidden_moves_masked():
    from gomoku.rules.renju import RenjuRules

    cfg = small_cfg(n_simulations=4, rule_set="renju", mcts_backend="cpp", patterns_backend="python")
    rules = RenjuRules()
    records = list(run_games(UniformEvaluator(), rules, cfg, 1, seed=3))
    assert len(records) == 1
    assert records[0].rule_set == "renju"

