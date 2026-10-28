from __future__ import annotations

import time

import numpy as np

from az.mcts import MCTS
from az.model import load_checkpoint
from config import DEFAULT_CONFIG
from gomoku.board import SIZE


class AZPlayer:
    """AlphaZero agent: PUCT search driven by a checkpoint, one leaf per inference call."""

    def __init__(self, checkpoint, rules, cfg: dict | None = None, time_budget_sec: float | None = None, seed: int | None = None, device=None) -> None:
        self.model, _ = load_checkpoint(checkpoint, device=device)
        self.rules = rules
        self.cfg = {**DEFAULT_CONFIG, **(cfg or {})}
        self.time_budget_sec = time_budget_sec
        self.rng = np.random.default_rng(seed)

    def select_move(self, board) -> tuple[int, int]:
        """Deterministic play (batch = 1) until n_simulations or the time budget runs out."""
        mcts = MCTS(
            rules=self.rules,
            n_simulations=int(self.cfg["n_simulations"]),
            c_puct=float(self.cfg["c_puct"]),
            dirichlet_alpha=float(self.cfg["dirichlet_alpha"]),
            dirichlet_eps=float(self.cfg["dirichlet_eps"]),
            root_noise=False,
            rng=self.rng,
        )
        mcts.reset(board)
        deadline = None if self.time_budget_sec is None else time.monotonic() + float(self.time_budget_sec)

        while deadline is None or time.monotonic() < deadline:
            request = mcts.request()
            if request is None:
                break
            probs, values = self.model.predict(request.planes[np.newaxis, ...])
            mcts.receive(probs[0], float(values[0]))

        if int(mcts.visit_counts().sum()) == 0:
            # Budget ran out before the first leaf: still owed a legal move.
            return divmod(int(self.rng.choice(self.rules.legal_moves(board))), SIZE)
        return divmod(int(mcts.select_move(0.0)), SIZE)
