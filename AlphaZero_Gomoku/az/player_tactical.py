from __future__ import annotations

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

    def select_move(self, board) -> tuple[int, int]:
        self.last_override_applied = False
        self.last_override_reason = None
        if not self.enabled:
            return self._fallback_move(board)

        legal = self.rules.legal_moves(board)
        player = board.current_player
        opponent = WHITE if player == BLACK else BLACK

        for move in legal:
            if self._wins_after(board, move, player):
                return self._override(move, "win_now")

        for move in legal:
            if self._wins_after(board, move, opponent):
                return self._override(move, "block_win")

        return self._fallback_move(board)

    def _fallback_move(self, board) -> tuple[int, int]:
        return self.base.select_move(board)

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
