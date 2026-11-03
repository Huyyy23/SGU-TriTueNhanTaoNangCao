from __future__ import annotations

import random

import pytest

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules import get_rules
from gomoku.runner import REASON_DRAW, REASON_WIN, Runner, render_board
from players.base import IllegalMoveError, Player
from players.human import HumanPlayer
from players.random import RandomPlayer


class ScriptedPlayer(Player):
    def __init__(self, moves):
        self.moves = iter(moves)

    def select_move(self, board):
        return next(self.moves)


def test_human_player_retries_invalid_input():
    answers = iter(("abc", "99,99", "5,5"))
    hints = []

    def prompt(_board, hint):
        hints.append(hint)
        return next(answers)

    player = HumanPlayer(get_rules("normal"), prompt)
    assert player.select_move(Board()) == (5, 5)
    assert hints == [None, "enter coordinates as row,col", "coordinates must be between 0 and 10"]


def test_human_player_accepts_supported_coordinate_formats():
    rules = get_rules("normal")
    for answer in ("3,4", "3 4", "3-4"):
        assert HumanPlayer(rules, lambda _board, _hint: answer).select_move(Board()) == (3, 4)


def test_runner_detects_win_and_reports_moves():
    rules = get_rules("normal")
    black_moves = [(5, col) for col in range(3, 8)]
    white_moves = [(0, col) for col in range(5)]
    result = Runner(rules).play(
        ScriptedPlayer(black_moves),
        ScriptedPlayer(white_moves),
    )

    assert result.reason == REASON_WIN
    assert result.winner == BLACK
    assert result.moves == [
        move
        for pair in zip(black_moves, white_moves)
        for move in pair
    ][:8] + [black_moves[-1]]


def test_runner_random_game_terminates():
    rules = get_rules("normal")
    rng = random.Random(12)
    result = Runner(rules).play(
        RandomPlayer(rules, seed=rng.randrange(1_000_000)),
        RandomPlayer(rules, seed=rng.randrange(1_000_000)),
    )

    assert result.reason in (REASON_WIN, REASON_DRAW)
    assert len(result.moves) == len(set(result.moves))
    assert len(result.moves) <= SIZE * SIZE


def test_runner_rejects_illegal_player_move():
    class BadPlayer(Player):
        def select_move(self, board):
            return (SIZE, 0)

    with pytest.raises(IllegalMoveError, match="out of range"):
        Runner(get_rules("normal")).play(BadPlayer(), RandomPlayer(get_rules("normal")))


def test_render_board_uses_11_by_11_grid():
    board = Board()
    board.play(5, 5)
    lines = render_board(board).splitlines()
    assert len(lines) == SIZE + 1
    assert "X" in lines[6]


def test_random_player_does_not_move_on_terminal_board():
    rules = get_rules("normal")
    board = Board()
    for col in range(5):
        board.grid[5 * SIZE + col] = BLACK
    with pytest.raises(RuntimeError):
        RandomPlayer(rules).select_move(board)
