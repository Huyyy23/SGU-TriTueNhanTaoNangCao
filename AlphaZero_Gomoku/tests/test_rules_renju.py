from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.features import board_to_planes
from gomoku.rules import RenjuRules, get_rules


def position(black=(), white=()):
    board = Board()
    for row, col in black:
        board.grid[row * SIZE + col] = BLACK
    for row, col in white:
        board.grid[row * SIZE + col] = WHITE
    return board


def test_renju_is_available_through_rules_factory():
    assert isinstance(get_rules("renju"), RenjuRules)


def test_black_overline_is_forbidden():
    row = SIZE // 2
    board = position(black=[(row, col) for col in range(2, 7)])

    assert row * SIZE + 7 in RenjuRules().forbidden_moves(board)


def test_black_double_four_is_forbidden():
    black = [
        (5, 3), (5, 4), (5, 6),
        (3, 5), (4, 5), (6, 5),
    ]
    board = position(black=black)

    assert 5 * SIZE + 5 in RenjuRules().forbidden_moves(board)


def test_black_double_three_is_forbidden_and_masked_from_legal_moves():
    board = position(black=[(5, 4), (5, 6), (4, 5), (6, 5)])
    rules = RenjuRules()
    move = 5 * SIZE + 5

    assert move in rules.forbidden_moves(board)
    assert move not in rules.legal_moves(board)


def test_exact_five_is_legal_and_wins():
    row = SIZE // 2
    board = position(black=[(row, col) for col in range(2, 6)])
    rules = RenjuRules()
    move = row * SIZE + 6

    assert move not in rules.forbidden_moves(board)
    board.play(row, 6)
    assert rules.is_win(board)


def test_exact_five_exception_overrides_other_forbidden_patterns():
    black = [(5, col) for col in range(1, 5)]
    black.extend(((3, 5), (4, 5), (6, 5), (3, 3), (4, 4), (6, 6)))
    board = position(black=black)

    assert 5 * SIZE + 5 not in RenjuRules().forbidden_moves(board)


def test_exact_five_exception_overrides_overline_in_another_direction():
    black = [(5, col) for col in range(2, 6)]
    black.extend((row, 6) for row in (2, 3, 4, 6, 7))
    board = position(black=black)

    assert 5 * SIZE + 6 not in RenjuRules().forbidden_moves(board)


def test_white_has_no_forbidden_moves_and_wins_with_overline():
    row = SIZE // 2
    board = position(white=[(row, col) for col in range(2, 8)])
    board.current_player = WHITE
    rules = RenjuRules()

    assert rules.forbidden_moves(board) == []
    assert rules.is_win(board)


def test_black_overline_is_not_a_win():
    row = SIZE // 2
    board = position(black=[(row, col) for col in range(2, 8)])

    assert not RenjuRules().is_win(board)


def test_forbidden_moves_cache_tracks_the_board_position():
    board = position(black=[(5, 4), (5, 6), (4, 5), (6, 5)])
    rules = RenjuRules()
    center = 5 * SIZE + 5

    assert center in rules.forbidden_moves(board)
    assert center in rules.forbidden_moves(board)
    board.play(5, 5)
    board.play(0, 0)
    assert center not in rules.forbidden_moves(board)


def test_renju_forbidden_feature_plane_marks_double_three():
    board = position(black=[(5, 4), (5, 6), (4, 5), (6, 5)])
    planes = board_to_planes(board, RenjuRules(), BLACK)

    assert planes[2, 5, 5] == 1
    assert planes.dtype.name == "uint8"


def test_forbidden_move_generation_does_not_mutate_board():
    board = position(black=[(5, 4), (5, 6), (4, 5), (6, 5)])
    original = board.grid.copy()

    RenjuRules().forbidden_moves(board)

    assert (board.grid == original).all()
    assert board.history == []


def test_full_board_with_no_winner_is_terminal_draw():
    board = Board()
    pattern = (BLACK, BLACK, WHITE, WHITE)
    board.grid[:] = [
        pattern[(row + 2 * col) % len(pattern)]
        for row in range(SIZE)
        for col in range(SIZE)
    ]
    rules = RenjuRules()

    assert rules.is_terminal(board)
    assert not rules.is_win(board)
    assert rules.legal_moves(board) == []
    assert EMPTY == 0
