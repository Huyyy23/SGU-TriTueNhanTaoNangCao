from __future__ import annotations

import re
from collections.abc import Callable

from gomoku.board import SIZE
from players.base import Player

InputCallback = Callable[[object, str | None], str | None]
_HYPHEN_SEPARATOR = re.compile(r"(?<=\d)-(?=\d)")


class HumanPlayer(Player):
    """Read coordinates through a callback and retry until a legal move is entered."""

    name = "human"

    def __init__(self, rules, prompt: InputCallback) -> None:
        self.rules = rules
        self.prompt = prompt

    def select_move(self, board) -> tuple[int, int]:
        hint = None
        while True:
            raw = self.prompt(board, hint)
            if raw is None:
                raise EOFError("no input available for HumanPlayer")
            try:
                row, col = _parse_move(raw)
            except ValueError as exc:
                hint = str(exc)
                continue
            if row * SIZE + col not in self.rules.legal_moves(board):
                hint = f"({row}, {col}) is not a legal move"
                continue
            return row, col


def _parse_move(raw: str) -> tuple[int, int]:
    cleaned = raw.strip()
    chunks = cleaned.split()
    if len(chunks) == 2:
        cleaned = ",".join(chunks)
    cleaned = _HYPHEN_SEPARATOR.sub(",", cleaned)
    parts = cleaned.split(",")
    if len(parts) != 2:
        raise ValueError("enter coordinates as row,col")
    try:
        row, col = (int(part.strip()) for part in parts)
    except ValueError:
        raise ValueError("coordinates must be integers") from None
    if not (0 <= row < SIZE and 0 <= col < SIZE):
        raise ValueError(f"coordinates must be between 0 and {SIZE - 1}")
    return row, col
