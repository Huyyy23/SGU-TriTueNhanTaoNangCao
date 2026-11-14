from __future__ import annotations

import math
import time

import numpy as np

from az.mcts import MCTS
from az.model import load_checkpoint
from config import DEFAULT_CONFIG
from gomoku.board import SIZE


class AZPlayer:
    """AlphaZero agent: PUCT search driven by a checkpoint, one leaf per inference call."""

    def __init__(self, checkpoint, rules, cfg: dict | None = None, time_budget_sec: float | None = None, seed: int | None = None, device=None) -> None:
        if time_budget_sec is not None and (not math.isfinite(time_budget_sec) or time_budget_sec <= 0):
            raise ValueError("time_budget_sec must be finite and positive")
        self.model, _ = load_checkpoint(checkpoint, device=device)
        self.rules = rules
        self.cfg = {**DEFAULT_CONFIG, **(cfg or {})}
        self.time_budget_sec = time_budget_sec
        self.rng = np.random.default_rng(seed)
        self.nn_calls = 0
        self.simulations_run = 0
        self.nodes_visited = 0
        self.last_override_applied = False
        self.last_override_reason = None

    def select_move(self, board) -> tuple[int, int]:
        """Use fixed simulations, or run complete simulations until the deadline."""
        self.nn_calls = 0
        self.simulations_run = 0
        self.nodes_visited = 0
        self.last_override_applied = False
        self.last_override_reason = None
        deadline = None if self.time_budget_sec is None else time.monotonic() + float(self.time_budget_sec)
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

        while deadline is None or time.monotonic() < deadline:
            if deadline is not None:
                # One simulation per request keeps terminal-only paths deadline bounded.
                mcts.n_simulations = mcts.simulations_done + 1
            request = mcts.request()
            if request is None:
                if deadline is None or mcts.root.is_terminal:
                    break
                continue
            probs, values = self.model.predict(request.planes[np.newaxis, ...])
            mcts.receive(probs[0], float(values[0]))
        self.nn_calls = mcts.nn_calls
        self.simulations_run = mcts.simulations_done
        # MCTS exposes simulation counts, not terminal-path node visits.
        self.nodes_visited = None if mcts.simulations_done else 0

        if int(mcts.visit_counts().sum()) == 0:
            # Budget ran out before the first leaf: still owed a legal move.
            return divmod(int(self.rng.choice(self.rules.legal_moves(board))), SIZE)
        return divmod(int(mcts.select_move(0.0)), SIZE)
