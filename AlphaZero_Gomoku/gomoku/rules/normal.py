from __future__ import annotations

import numpy as np

from gomoku.board import EMPTY, SIZE
from gomoku.rules.base import Rules

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class NormalRules(Rules):
    """Five or more in a row wins, for both colours."""

    name = "normal"

    def legal_moves(self, board) -> list[int]:
        return np.flatnonzero(board.grid == EMPTY).tolist()

    @staticmethod
    def _run_length(grid: np.ndarray, r: int, c: int, dr: int, dc: int, player: int) -> int:
        count = 1
        for sign in (-1, 1):
            rr, cc = r + sign * dr, c + sign * dc
            while 0 <= rr < SIZE and 0 <= cc < SIZE and int(grid[rr, cc]) == player:
                count += 1
                rr += sign * dr
                cc += sign * dc
        return count

    def is_win(self, board) -> bool:
        if not board.history:
            grid = board.grid.reshape(SIZE, SIZE)
            for r, c in np.argwhere(grid != EMPTY):
                player = int(grid[r, c])
                if any(
                    self._run_length(grid, int(r), int(c), dr, dc, player) >= 5
                    for dr, dc in _DIRECTIONS
                ):
                    return True
            return False

        r, c = divmod(board.history[-1], SIZE)
        player = int(board.grid[r * SIZE + c])
        grid = board.grid.reshape(SIZE, SIZE)
        return any(
            self._run_length(grid, r, c, dr, dc, player) >= 5
            for dr, dc in _DIRECTIONS
        )
