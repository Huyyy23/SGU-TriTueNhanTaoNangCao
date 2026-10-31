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
    assert record.num_moves == 9


def test_full_board_is_a_draw() -> None:
    class FirstLegal:
        name = "first"

        def select_move(self, board) -> tuple[int, int]:
            return divmod(int(np.flatnonzero(board.grid == 0)[0]), 11)

    rules = NoWinRules()
    record = play_game(FirstLegal(), FirstLegal(), rules, game_id=0)

    assert record.winner == "draw"
    assert record.end_reason == "board_full"
    assert record.num_moves == 121


def test_colours_alternate_and_summary_adds_up() -> None:
    rules = NormalRules()
    a = RandomPlayer(rules, seed=1)
    b = RandomPlayer(rules, seed=2)
    b.name = "random_b"

    records = evaluate(a, b, rules, 10)

    assert [r.game_id for r in records] == list(range(10))
    assert sum(1 for r in records if r.black_player == "random") == 5
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
