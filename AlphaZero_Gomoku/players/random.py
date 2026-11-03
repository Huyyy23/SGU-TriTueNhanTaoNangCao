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
        legal_moves = self.rules.legal_moves(board)
        if not legal_moves or self.rules.is_win(board):
            raise RuntimeError("RandomPlayer asked to move on a terminal board")
        return divmod(int(self.rng.choice(legal_moves)), SIZE)
