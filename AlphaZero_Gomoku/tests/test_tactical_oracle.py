import numpy as np
import pytest

from gomoku.board import Board, SIZE
from gomoku.rules.normal import NormalRules


# Five-cell windows are independent of the production win detector.
WINDOWS = [
    tuple((r + k * dr) * SIZE + c + k * dc for k in range(5))
    for r in range(SIZE)
    for c in range(SIZE)
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1))
    if 0 <= r + 4 * dr < SIZE and 0 <= c + 4 * dc < SIZE
]


def winning_moves(grid, player):
    result = set()
    for window in WINDOWS:
        values = grid[list(window)]
        if np.count_nonzero(values == player) == 4 and np.count_nonzero(values == 0) == 1:
            result.add(window[int(np.flatnonzero(values == 0)[0])])
    return result


def forks(grid, player):
    result = {}
    for action in np.flatnonzero(grid == 0):
        grid[action] = player
        wins = winning_moves(grid, player)
        grid[action] = 0
        if len(wins) >= 2:
            result[int(action)] = wins
    return result


@pytest.mark.parametrize(
    ('moves', 'bad_move', 'fork_move', 'expected_wins'),
    [
        ([20,72,31,58,61,60,9,48,42,53,84,38,28,49,71,50,51,41,81,91,83,82,30,29], 52, 17, {5,65}),
        ([20,72,31,58,42,53,9,18,60,70,49,27,71,82,46,94,106,50,38,61,83,48,59,95,39], 96, 69, {29,79}),
    ],
)
def test_recorded_loss_fork_has_two_independent_winning_replies(moves, bad_move, fork_move, expected_wins):
    board = Board()
    for action in moves:
        board.play(*divmod(action, SIZE))
    defender = board.current_player
    attacker = 3 - defender
    assert not winning_moves(board.grid, defender)
    assert not winning_moves(board.grid, attacker)
    assert fork_move in forks(board.grid, attacker)

    board.play(*divmod(bad_move, SIZE))
    board.play(*divmod(fork_move, SIZE))
    assert winning_moves(board.grid, attacker) == expected_wins
    assert not winning_moves(board.grid, defender)

    # Even after blocking either square, the other reply wins under real rules.
    rules = NormalRules()
    for block in expected_wins:
        board.play(*divmod(block, SIZE))
        reply = (expected_wins - {block}).pop()
        board.play(*divmod(reply, SIZE))
        assert rules.is_win(board)
        board.undo()
        board.undo()
