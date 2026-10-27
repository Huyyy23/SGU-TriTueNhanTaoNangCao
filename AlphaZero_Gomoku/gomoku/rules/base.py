from __future__ import annotations

from gomoku.board import EMPTY


class Rules:
    """Rules interface. MCTS, players and UI call only these methods."""

    name: str = "normal"

    def legal_moves(self, board) -> list[int]:
        """Flat indices of the empty cells."""
        raise NotImplementedError

    def forbidden_moves(self, board) -> list[int]:
        """Flat indices that the player to move may not play. Empty for Normal."""
        return []

    def is_win(self, board) -> bool:
        raise NotImplementedError

    def is_terminal(self, board) -> bool:
        return self.is_win(board) or not bool((board.grid == EMPTY).any())
