from __future__ import annotations

import time

from gomoku.board import BLACK, SIZE, WHITE


class TacticalAZPlayer:
    """AZ player with a shallow tactical override (win now, block opponent win).

    The base player is untouched; this wrapper only replaces its move when a
    one-move win or a one-move block exists. Candidates come from
    `rules.legal_moves`, so Renju forbidden moves are never chosen.
    """

    def __init__(self, base_player, enabled: bool = False, cfg: dict | None = None) -> None:
        self.base = base_player
        self.rules = base_player.rules
        tactical = (cfg or {}).get("tactical") or {}
        self.enabled = bool(tactical.get("enabled", enabled))
        self.last_override_applied = False
        self.last_override_reason: str | None = None
        self.nn_calls = 0
        self.simulations_run = 0
        self.nodes_visited = 0

    def select_move(self, board) -> tuple[int, int]:
        self.last_override_applied = False
        self.last_override_reason = None
        self.nn_calls = 0
        self.simulations_run = 0
        self.nodes_visited = 0
        budget = getattr(self.base, "time_budget_sec", None)
        self._deadline = None if budget is None else time.monotonic() + budget
        if not self.enabled:
            return self._fallback_move(board)

        legal = self.rules.legal_moves(board)
        player = board.current_player
        opponent = WHITE if player == BLACK else BLACK

        for move in legal:
            if self._deadline is not None and time.monotonic() >= self._deadline:
                return self._fallback_move(board)
            if self._wins_after(board, move, player):
                return self._override(move, "win_now")

        saved = board.current_player
        board.current_player = opponent
        try:
            opponent_legal = set(self.rules.legal_moves(board))
        finally:
            board.current_player = saved
        for move in legal:
            if self._deadline is not None and time.monotonic() >= self._deadline:
                return self._fallback_move(board)
            if move in opponent_legal and self._wins_after(board, move, opponent):
                return self._override(move, "block_win")

        return self._fallback_move(board)

    def _fallback_move(self, board) -> tuple[int, int]:
        budget = getattr(self.base, "time_budget_sec", None)
        if self._deadline is not None:
            self.base.time_budget_sec = max(0.0, self._deadline - time.monotonic())
        try:
            return self.base.select_move(board)
        finally:
            if self._deadline is not None:
                self.base.time_budget_sec = budget
            self.nn_calls = getattr(self.base, "nn_calls", 0)
            self.simulations_run = getattr(self.base, "simulations_run", 0)
            nodes = getattr(self.base, "nodes_visited", 0)
            self.nodes_visited = None if nodes is None else self.nodes_visited + nodes

    def _override(self, move: int, reason: str) -> tuple[int, int]:
        self.last_override_applied = True
        self.last_override_reason = reason
        return divmod(move, SIZE)

    def _wins_after(self, board, move: int, player: int) -> bool:
        """Play `move` as `player` on a copy-free path, then restore the board."""
        saved = board.current_player
        board.current_player = player
        board.play(*divmod(move, SIZE))
        try:
            return self.rules.is_win(board)
        finally:
            board.undo()
            board.current_player = saved
