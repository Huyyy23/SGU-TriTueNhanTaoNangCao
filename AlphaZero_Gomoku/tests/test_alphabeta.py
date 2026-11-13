from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.rules import get_rules
from players.alphabeta import AlphaBetaPlayer


def test_alphabeta_takes_immediate_win():
    board = Board()
    for col in range(2, 6):
        board.grid[5 * SIZE + col] = BLACK

    assert AlphaBetaPlayer(get_rules("normal"), depth=1).select_move(board) in ((5, 1), (5, 6))


def test_pruning_and_full_minimax_return_same_move_and_visit_counts():
    board = Board()
    for row in range(SIZE):
        for col in range(SIZE):
            board.grid[row * SIZE + col] = (
                BLACK if (row + 2 * col) % 5 < 2 else WHITE
            )
    for row, col in ((2, 2), (2, 8), (5, 5), (5, 6), (8, 2), (8, 8)):
        board.grid[row * SIZE + col] = 0
    rules = get_rules("normal")
    pruned = AlphaBetaPlayer(rules, depth=2, pruning=True)
    plain = AlphaBetaPlayer(rules, depth=2, pruning=False)

    assert pruned.select_move(board) == plain.select_move(board)
    assert pruned.nodes_visited < plain.nodes_visited
    assert board.history == []
