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
