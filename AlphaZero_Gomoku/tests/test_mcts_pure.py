import random

from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.rules import RenjuRules, get_rules
from players.mcts_pure import MCTSPure


def test_pure_mcts_takes_immediate_winning_move():
    board = Board()
    for col in (2, 3, 4, 5):
        board.grid[5 * SIZE + col] = BLACK
    player = MCTSPure(get_rules("normal"), n_rollouts=2, rng=random.Random(1))

    assert player.select_move(board) in ((5, 1), (5, 6))
    assert board.history == []


def test_pure_mcts_blocks_opponents_immediate_win():
    board = Board()
    for col in (2, 3, 4, 5):
        board.grid[5 * SIZE + col] = BLACK
    board.current_player = WHITE
    player = MCTSPure(get_rules("normal"), n_rollouts=2, rng=random.Random(1))

    assert player.select_move(board) in ((5, 1), (5, 6))
    assert board.history == []


def test_pure_mcts_only_selects_legal_renju_moves():
    board = Board()
    for row, col in ((5, 4), (5, 6), (4, 5), (6, 5)):
        board.grid[row * SIZE + col] = BLACK
    rules = RenjuRules()
    player = MCTSPure(
        rules,
        n_rollouts=1,
        max_playouts=1,
        rng=random.Random(4),
    )

    row, col = player.select_move(board)
    assert row * SIZE + col not in rules.forbidden_moves(board)
