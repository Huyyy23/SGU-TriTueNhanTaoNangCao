from __future__ import annotations

import numpy as np

from gomoku.board import SIZE
from players.base import Player


class RandomPlayer(Player):
    """Uniform random legal move. Baseline opponent, not a deliverable agent."""

    name = "random"

    def __init__(self, rules, seed: int | None = None) -> None:
        self.rules = rules
        self.rng = np.random.default_rng(seed)

    def select_move(self, board) -> tuple[int, int]:
        return divmod(int(self.rng.choice(self.rules.legal_moves(board))), SIZE)
