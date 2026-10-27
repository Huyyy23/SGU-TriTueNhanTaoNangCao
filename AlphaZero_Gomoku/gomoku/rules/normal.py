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

    def is_win(self, board) -> bool:
        grid = board.grid.reshape(SIZE, SIZE)
        padded = np.pad(grid, 4)
        center = padded[4 : 4 + SIZE, 4 : 4 + SIZE]
        for dr, dc in _DIRECTIONS:
            run = center != EMPTY
            for k in range(1, 5):
                shifted = padded[4 + dr * k : 4 + dr * k + SIZE, 4 + dc * k : 4 + dc * k + SIZE]
                run = run & (shifted == center)
                if not run.any():
                    break
            if run.any():
                return True
        return False
