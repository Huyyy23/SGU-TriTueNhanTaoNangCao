from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.rules import RenjuRules, get_rules
from players.heuristic import HeuristicPlayer


def test_heuristic_takes_win_before_blocking():
    board = Board()
    for col in range(2, 6):
        board.grid[5 * SIZE + col] = BLACK
    board.current_player = WHITE
    for col in range(3, 7):
        board.grid[6 * SIZE + col] = WHITE

    assert HeuristicPlayer(get_rules("normal")).select_move(board) in ((6, 2), (6, 7))


def test_heuristic_blocks_immediate_win():
    board = Board()
    for col in range(2, 6):
        board.grid[5 * SIZE + col] = BLACK
    board.current_player = WHITE

    assert HeuristicPlayer(get_rules("normal")).select_move(board) in ((5, 1), (5, 6))


def test_heuristic_never_selects_renju_forbidden_move():
    board = Board()
    for r, c in ((5, 4), (5, 6), (4, 5), (6, 5)):
        board.grid[r * SIZE + c] = BLACK
    rules = RenjuRules()

    r, c = HeuristicPlayer(rules).select_move(board)

    assert r * SIZE + c in rules.legal_moves(board)
