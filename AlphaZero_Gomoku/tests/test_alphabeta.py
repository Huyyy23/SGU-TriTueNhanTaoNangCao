from gomoku.board import BLACK, SIZE, Board
from gomoku.rules import get_rules
from players.alphabeta import AlphaBetaPlayer


def test_alphabeta_takes_immediate_win():
    board = Board()
    for col in range(2, 6):
        board.grid[5 * SIZE + col] = BLACK

    assert AlphaBetaPlayer(get_rules("normal"), depth=1).select_move(board) in ((5, 1), (5, 6))


def test_pruning_and_full_minimax_return_same_move_and_visit_counts():
    board = Board()
    for r, c, player in ((5, 5, BLACK), (5, 6, 2), (4, 5, BLACK), (6, 6, 2)):
        board.grid[r * SIZE + c] = player
    rules = get_rules("normal")
    pruned = AlphaBetaPlayer(rules, depth=2, pruning=True)
    plain = AlphaBetaPlayer(rules, depth=2, pruning=False)

    assert pruned.select_move(board) == plain.select_move(board)
    assert pruned.nodes_visited < plain.nodes_visited
    assert board.history == []
