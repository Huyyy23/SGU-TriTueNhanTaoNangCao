from __future__ import annotations


class Player:
    """Agent interface: pick one move for the side to move on `board`."""

    name: str = "player"

    def select_move(self, board) -> tuple[int, int]:
        raise NotImplementedError
