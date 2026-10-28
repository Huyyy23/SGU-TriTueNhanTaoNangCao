import numpy as np
import pytest

from az.replay_buffer import RULE_SET_IDS, ReplayBuffer
from az.selfplay import run_games
from config import DEFAULT_CONFIG
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


def test_one_game_sample_count_matches_moves():
    buffer = ReplayBuffer(capacity=1000)
    records = list(run_games(UniformEvaluator(), NormalRules(), small_cfg(), 1, seed=3, buffer=buffer))
    record = records[0]

    assert record.validate() is None
    assert record.num_moves == len(buffer) > 0
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
    assert (first.winner, first.num_moves, first.nn_calls) == (second.winner, second.num_moves, second.nn_calls)


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
    seconds_per_20_moves = record.duration_sec / record.num_moves * 20
    assert seconds_per_20_moves < 5.0, (
        f"{seconds_per_20_moves:.2f}s per 20 moves "
        f"({record.duration_sec:.2f}s for {record.num_moves} moves, {cfg['n_simulations']} sims)"
    )


def test_parallel_games_batches_leaves_and_keeps_search_budget_per_move():
    single = list(run_games(UniformEvaluator(), NormalRules(), small_cfg(parallel_games=1), 3, seed=3))
    batched = list(run_games(UniformEvaluator(), NormalRules(), small_cfg(parallel_games=4), 3, seed=3))

    assert all(record.mean_batch == 1.0 for record in single)
    assert all(record.mean_batch > 1.0 for record in batched)

    ratio_single = sum(r.nn_calls / r.num_moves for r in single) / len(single)
    ratio_batched = sum(r.nn_calls / r.num_moves for r in batched) / len(batched)
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

