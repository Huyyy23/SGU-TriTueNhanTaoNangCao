from __future__ import annotations

import numpy as np

from az.evaluator import evaluate, play_game, summarise, wilson_interval
from gomoku.rules.normal import NormalRules
from players.random import RandomPlayer


class ScriptedPlayer:
    """Plays a fixed list of cells, one per turn."""

    name = "scripted"

    def __init__(self, moves) -> None:
        self.moves = iter(moves)

    def select_move(self, board) -> tuple[int, int]:
        return next(self.moves)


class NoWinRules:
    """Fake rules that only stop on a full board, to exercise the draw branch."""

    name = "fake"

    def legal_moves(self, board) -> list[int]:
        return np.flatnonzero(board.grid == 0).tolist()

    def is_terminal(self, board) -> bool:
        return not bool((board.grid == 0).any())

    def is_win(self, board) -> bool:
        return False


def test_win_is_recorded_as_five_in_row() -> None:
    black = ScriptedPlayer([(5, k) for k in range(5)])
    white = ScriptedPlayer([(0, k) for k in range(4)])

    record = play_game(black, white, NormalRules(), game_id=0)

    assert record.winner == "black"
    assert record.first_player_won is True
    assert record.end_reason == "five_in_row"
    assert record.moves_count == 9


def test_full_board_is_a_draw() -> None:
    class FirstLegal:
        name = "first"

        def select_move(self, board) -> tuple[int, int]:
            return divmod(int(np.flatnonzero(board.grid == 0)[0]), 11)

    rules = NoWinRules()
    record = play_game(FirstLegal(), FirstLegal(), rules, game_id=0)

    assert record.winner == "draw"
    assert record.end_reason == "board_full"
    assert record.moves_count == 121


def test_colours_alternate_and_summary_adds_up() -> None:
    rules = NormalRules()
    a = RandomPlayer(rules, seed=1)
    b = RandomPlayer(rules, seed=2)
    b.name = "random_b"

    records = evaluate(a, b, rules, 10)

    assert [r.game_id for r in records] == list(range(10))
    assert sum(1 for r in records if r.first_player == "random") == 5
    summary = summarise(records, "random", "random_b")
    assert summary["total_games"] == 10
    assert summary["as_black"]["games"] + summary["as_white"]["games"] == 10
    low, high = summary["wilson_95_a"]
    assert 0.0 <= low <= summary["overall_win_rate_a"] <= high <= 1.0


def test_wilson_interval_narrows_with_more_games() -> None:
    low_100, high_100 = wilson_interval(50, 100)
    low_1000, high_1000 = wilson_interval(500, 1000)

    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert low_100 < 0.5 < high_100
    assert (high_1000 - low_1000) < (high_100 - low_100)


def test_raw_move_telemetry_and_draw_score():
    from dataclasses import replace

    black = ScriptedPlayer([(5, k) for k in range(5)])
    white = ScriptedPlayer([(0, k) for k in range(4)])
    black.name, white.name = "a", "b"
    black.nodes_visited = 7
    black.simulations_run = 3
    record = play_game(black, white, NormalRules(), 0)
    assert len(record.moves) == record.moves_count == 9
    assert all(move.latency_sec >= 0 for move in record.moves)
    assert record.moves[0].nodes_visited == 7
    assert record.moves[0].simulations_run == 3
    summary = summarise([record, replace(record, winner="draw", first_player_won=False)], "a", "b")
    assert (summary["wins"], summary["draws"], summary["losses"]) == (1, 1, 0)
    assert summary["score"] == .75
    assert summary["as_black"]["draws"] == 1
    assert summary["telemetry"]["a"]["nodes_visited"] == 70
    assert summary["telemetry"]["a"]["simulations_run"] == 30
    assert summary["telemetry"]["b"]["nodes_visited"] is None
    assert summary["score_ci_95"] == wilson_interval(1.5, 2)


def test_neural_calls_and_override_rates_are_recorded():
    black = ScriptedPlayer([(5, k) for k in range(5)])
    white = ScriptedPlayer([(0, k) for k in range(4)])
    black.name, white.name = "a", "b"
    black.nn_calls = 3
    black.last_override_applied = True
    black.last_override_reason = "win_now"
    record = play_game(black, white, NormalRules(), 0)
    assert record.nn_calls == 15
    assert record.moves[0].nn_calls == 3
    assert record.moves[0].override_reason == "win_now"
    summary = summarise([record], "a", "b")
    assert summary["telemetry"]["a"]["nn_calls"] == 15
    assert summary["telemetry"]["a"]["override_count"] == 5
    assert summary["telemetry"]["a"]["override_rate"] == 1.0
    assert summary["telemetry"]["b"]["override_rate"] == 0.0


def test_invalid_moves_rejected_without_board_mutation():
    import pytest

    class InvalidPlayer:
        name = "invalid"

        def __init__(self, move):
            self.move = move

        def select_move(self, board):
            self.board = board
            return self.move

    class RestrictedRules(NormalRules):
        name = "restricted"

        def legal_moves(self, board):
            return [move for move in super().legal_moves(board) if move != 60]

    for move in [(-1, 0), (11, 0), (0.5, 0), (5, 5)]:
        player = InvalidPlayer(move)
        with pytest.raises(ValueError, match="illegal move"):
            play_game(player, player, RestrictedRules(), 0)
        assert player.board.history == []
        assert not player.board.grid.any()


def test_renju_overline_is_rejected_before_play():
    import pytest
    from gomoku.rules.renju import RenjuRules

    black = ScriptedPlayer([(5, col) for col in (0, 1, 2, 4, 5, 3)])
    white = ScriptedPlayer([(0, col) for col in (0, 2, 4, 6, 8)])
    with pytest.raises(ValueError, match="illegal move"):
        play_game(black, white, RenjuRules(), 0)
