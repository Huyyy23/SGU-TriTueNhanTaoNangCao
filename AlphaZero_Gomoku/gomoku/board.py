from __future__ import annotations

import numpy as np

SIZE = 11
EMPTY, BLACK, WHITE = 0, 1, 2


class Board:
    """11x11 Gomoku board state. Holds no rules."""

    def __init__(self) -> None:
        self.grid = np.zeros(SIZE * SIZE, dtype=np.int8)
        self.current_player: int = BLACK
        self.history: list[int] = []

    def play(self, r: int, c: int) -> int:
        """Place a stone for the player to move and return its flat index."""
        if not (0 <= r < SIZE and 0 <= c < SIZE):
            raise ValueError(f"Cell ({r}, {c}) is outside the board.")
        idx = r * SIZE + c
        if self.grid[idx] != EMPTY:
            raise ValueError(f"Cell ({r}, {c}) is already occupied.")
        self.grid[idx] = self.current_player
        self.history.append(idx)
        self.current_player = WHITE if self.current_player == BLACK else BLACK
        return idx

    def undo(self) -> tuple[int, int]:
        """Remove the last stone and return the cell it occupied as (row, col)."""
        if not self.history:
            raise ValueError("No moves to undo.")
        idx = self.history.pop()
        self.grid[idx] = EMPTY
        self.current_player = WHITE if self.current_player == BLACK else BLACK
        return divmod(idx, SIZE)
