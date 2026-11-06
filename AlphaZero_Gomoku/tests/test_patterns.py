import pytest

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.patterns import find_patterns


def board_with(stones: dict[tuple[int, int], int]) -> Board:
    board = Board()
    for (r, c), player in stones.items():
        board.grid[r * SIZE + c] = player
    return board


def test_finds_open_three_and_reports_empty_ends():
    board = board_with({(5, c): BLACK for c in (4, 5)})
    before = board.grid.copy()

    patterns = find_patterns(board, 5, 6, BLACK)

    assert [(p.kind, p.direction, p.completion_cells) for p in patterns] == [
        ("open_three", (0, 1), ((5, 3), (5, 7)))
    ]
    assert (board.grid == before).all()


def test_finds_open_four_in_all_four_directions():
    cases = [
        ((0, 1), (5, 4), [(5, c) for c in (5, 6, 7)], (5, 8)),
        ((1, 0), (4, 5), [(r, 5) for r in (5, 6, 7)], (8, 5)),
        ((1, 1), (5, 5), [(r, r) for r in (3, 4, 6)], (7, 7)),
        ((1, -1), (5, 5), [(r, 10-r) for r in (3, 4, 6)], (7, 3)),
    ]
    for direction, candidate, stones, end in cases:
        board = board_with({cell: BLACK for cell in stones})
        pattern, = find_patterns(board, *candidate, BLACK)
        assert pattern.kind == "open_four"
        assert pattern.direction == direction
        assert end in pattern.completion_cells


def test_distinguishes_blocked_four_and_overline():
    blocked = board_with({(5, c): BLACK for c in (4, 5, 6)} | {(5, 3): WHITE})
    pattern, = find_patterns(blocked, 5, 7, BLACK)
    assert pattern.kind == "blocked_four"
    assert pattern.completion_cells == ((5, 8),)

    overline = board_with({(5, c): BLACK for c in (3, 4, 5, 6, 7)})
    pattern, = find_patterns(overline, 5, 8, BLACK)
    assert pattern.kind == "overline"


def test_finds_blocked_three():
    board = board_with({(5, 3): WHITE, (5, 4): BLACK, (5, 5): BLACK})
    pattern, = find_patterns(board, 5, 6, BLACK)
    assert (pattern.kind, pattern.completion_cells) == ("blocked_three", ((5, 7),))


def test_finds_broken_three_completion_gap_only():
    board = board_with({(5, 3): BLACK, (5, 6): BLACK})
    pattern, = find_patterns(board, 5, 4, BLACK)
    assert (pattern.kind, pattern.completion_cells) == ("broken_three", ((5, 5),))


def test_finds_new_patterns_for_both_players_and_all_directions():
    cases = [
        ((0, 1), (5, 6), {(5, 3): WHITE, (5, 4): BLACK, (5, 5): BLACK}, BLACK, "blocked_three"),
        ((1, 0), (6, 5), {(3, 5): WHITE, (4, 5): BLACK, (5, 5): BLACK}, BLACK, "blocked_three"),
        ((1, 1), (6, 6), {(3, 3): WHITE, (4, 4): BLACK, (5, 5): BLACK}, BLACK, "blocked_three"),
        ((1, -1), (6, 4), {(3, 7): WHITE, (4, 6): BLACK, (5, 5): BLACK}, BLACK, "blocked_three"),
        ((0, 1), (5, 4), {(5, 3): WHITE, (5, 6): WHITE}, WHITE, "broken_three"),
        ((1, 0), (4, 5), {(3, 5): WHITE, (6, 5): WHITE}, WHITE, "broken_three"),
        ((1, 1), (4, 4), {(3, 3): WHITE, (6, 6): WHITE}, WHITE, "broken_three"),
        ((1, -1), (4, 6), {(3, 7): WHITE, (6, 4): WHITE}, WHITE, "broken_three"),
    ]
    for direction, candidate, stones, player, kind in cases:
        board = board_with(stones)
        patterns = find_patterns(board, *candidate, player)
        matches = [p for p in patterns if p.kind == kind and p.direction == direction]
        assert matches
        if kind == "broken_three":
            gap = (candidate[0] + direction[0], candidate[1] + direction[1])
            assert matches[0].completion_cells == (gap,)


def test_ignores_disconnected_or_opponent_stones():
    board = board_with({(5, 2): BLACK, (5, 4): BLACK, (5, 5): WHITE})
    assert find_patterns(board, 5, 6, BLACK) == []


@pytest.mark.parametrize("row,col,player", [(-1, 0, BLACK), (0, SIZE, BLACK), (0, 0, EMPTY), (0, 0, 3)])
def test_rejects_invalid_arguments(row, col, player):
    with pytest.raises(ValueError):
        find_patterns(Board(), row, col, player)


def test_rejects_occupied_candidate():
    board = board_with({(5, 5): WHITE})
    with pytest.raises(ValueError):
        find_patterns(board, 5, 5, BLACK)
