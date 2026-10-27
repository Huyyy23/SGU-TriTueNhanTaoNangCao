import numpy as np

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules.normal import NormalRules


def board_with(stones: dict[tuple[int, int], int]) -> Board:
    """Fill cells directly; win checks only read the grid."""
    board = Board()
    for (r, c), player in stones.items():
        board.grid[r * SIZE + c] = player
    return board


def test_horizontal_win():
    rules = NormalRules()
    stones = {(5, c): BLACK for c in range(5)}
    assert rules.is_win(board_with(stones))


def test_vertical_win():
    rules = NormalRules()
    stones = {(r, 7): WHITE for r in range(3, 8)}
    assert rules.is_win(board_with(stones))


def test_diagonal_down_right_win():
    rules = NormalRules()
    stones = {(r, r): BLACK for r in range(2, 7)}
    assert rules.is_win(board_with(stones))


def test_diagonal_down_left_win():
    rules = NormalRules()
    stones = {(r, 10 - r): WHITE for r in range(0, 5)}
    assert rules.is_win(board_with(stones))


def test_six_in_a_row_wins():
    rules = NormalRules()
    stones = {(9, c): BLACK for c in range(1, 7)}
    assert rules.is_win(board_with(stones))


def test_four_in_a_row_is_not_a_win():
    rules = NormalRules()
    stones = {(5, c): BLACK for c in range(4)}
    assert not rules.is_win(board_with(stones))


def test_win_must_be_contiguous():
    rules = NormalRules()
    stones = {(5, 0): BLACK, (5, 1): BLACK, (5, 2): BLACK, (5, 4): BLACK, (5, 5): BLACK}
    assert not rules.is_win(board_with(stones))


def test_legal_moves_excludes_occupied():
    rules = NormalRules()
    board = Board()
    board.play(0, 0)
    board.play(5, 5)
    legal = rules.legal_moves(board)
    assert len(legal) == SIZE * SIZE - 2
    assert 0 not in legal
    assert 5 * SIZE + 5 not in legal
    assert 1 in legal


def test_forbidden_moves_empty_for_normal():
    assert NormalRules().forbidden_moves(Board()) == []


def test_full_board_is_terminal():
    board = Board()
    board.grid[:] = np.where(np.arange(SIZE * SIZE) % 2 == 0, BLACK, WHITE)
    rules = NormalRules()
    assert rules.legal_moves(board) == []
    assert rules.is_terminal(board) is True


def test_empty_board_is_not_terminal():
    assert NormalRules().is_terminal(Board()) is False
    assert (Board().grid == EMPTY).all()
