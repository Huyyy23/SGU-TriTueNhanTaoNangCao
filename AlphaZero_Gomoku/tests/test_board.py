import numpy as np
import pytest

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board


def test_initial_state():
    board = Board()
    assert board.grid.shape == (SIZE * SIZE,)
    assert board.grid.dtype == np.int8
    assert (board.grid == EMPTY).all()
    assert board.current_player == BLACK
    assert board.history == []


def test_play_places_stone_and_flips_turn():
    board = Board()
    idx = board.play(5, 6)
    assert idx == 5 * SIZE + 6
    assert board.grid[idx] == BLACK
    assert board.current_player == WHITE
    assert board.history == [idx]

    board.play(0, 0)
    assert board.grid[0] == WHITE
    assert board.current_player == BLACK


def test_play_out_of_range_raises():
    board = Board()
    for r, c in [(-1, 0), (0, -1), (SIZE, 0), (0, SIZE)]:
        with pytest.raises(ValueError):
            board.play(r, c)
    assert board.history == []


def test_play_occupied_raises():
    board = Board()
    board.play(3, 3)
    with pytest.raises(ValueError):
        board.play(3, 3)
    assert board.current_player == WHITE
    assert len(board.history) == 1


def test_undo_restores_state():
    board = Board()
    board.play(2, 7)
    board.play(4, 1)
    assert board.undo() == (4, 1)
    assert board.grid[4 * SIZE + 1] == EMPTY
    assert board.current_player == WHITE
    assert board.undo() == (2, 7)
    assert (board.grid == EMPTY).all()
    assert board.current_player == BLACK
    assert board.history == []


def test_undo_empty_raises():
    with pytest.raises(ValueError):
        Board().undo()
