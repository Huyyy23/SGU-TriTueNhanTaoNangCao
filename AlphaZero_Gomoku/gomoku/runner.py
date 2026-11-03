from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from players.base import IllegalMoveError, Player

REASON_WIN = "win"
REASON_DRAW = "draw"


@dataclass
class GameResult:
    winner: int
    reason: str
    moves: list[tuple[int, int]] = field(default_factory=list)

    @property
    def is_draw(self) -> bool:
        return self.winner == EMPTY


def render_board(board: Board) -> str:
    symbols = {EMPTY: ".", BLACK: "X", WHITE: "O"}
    header = "    " + " ".join(str(col) for col in range(SIZE))
    lines = [header]
    for row in range(SIZE):
        start = row * SIZE
        cells = " ".join(symbols[int(value)] for value in board.grid[start : start + SIZE])
        lines.append(f"{row:>2}  {cells}")
    return "\n".join(lines)


class Runner:
    """Run a complete game using only the Board, Player, and Rules interfaces."""

    def __init__(self, rules) -> None:
        self.rules = rules

    def play(
        self,
        black: Player,
        white: Player,
        *,
        on_move: Callable[[Board, Player, int], None] | None = None,
        max_moves: int | None = None,
    ) -> GameResult:
        board = Board()
        players = {BLACK: black, WHITE: white}
        moves: list[tuple[int, int]] = []

        while True:
            side = board.current_player
            mover = players[side]
            move = mover.select_move(board)
            row, col = self._validate_move(board, move, mover)
            board.play(row, col)
            moves.append((row, col))

            if on_move is not None:
                on_move(board, mover, side)

            if self.rules.is_win(board):
                return GameResult(winner=side, reason=REASON_WIN, moves=moves)
            if self.rules.is_terminal(board):
                return GameResult(winner=EMPTY, reason=REASON_DRAW, moves=moves)
            if max_moves is not None and len(moves) >= max_moves:
                raise RuntimeError(
                    f"game exceeded max_moves={max_moves} without reaching a terminal state"
                )

    def _validate_move(
        self, board: Board, move, player: Player
    ) -> tuple[int, int]:
        name = getattr(player, "name", type(player).__name__)
        if (
            not isinstance(move, tuple)
            or len(move) != 2
            or not all(isinstance(value, Integral) for value in move)
        ):
            raise IllegalMoveError(f"{name} returned malformed move {move!r}")

        row, col = map(int, move)
        if not (0 <= row < SIZE and 0 <= col < SIZE):
            raise IllegalMoveError(f"{name} moved out of range: ({row}, {col})")
        index = row * SIZE + col
        if index not in self.rules.legal_moves(board):
            raise IllegalMoveError(f"{name} returned illegal move ({row}, {col})")
        return row, col
