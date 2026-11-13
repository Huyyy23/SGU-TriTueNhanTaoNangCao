from __future__ import annotations

from numbers import Integral

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board
from gomoku.rules import Rules, get_rules
from gomoku.runner import render_board


class Game:
    """Thin stateful API over a Gomoku board and its selected rules."""

    def __init__(self, rules: Rules | str = "normal") -> None:
        self.rules = get_rules(rules) if isinstance(rules, str) else rules
        self.board = Board()

    def reset(self) -> None:
        self.board = Board()

    def legal_actions(self) -> list[int]:
        return self.rules.legal_moves(self.board)

    def step(self, action: int) -> None:
        if not isinstance(action, Integral) or isinstance(action, bool):
            raise ValueError(f"action must be an integer board index, got {action!r}")
        action = int(action)
        if not 0 <= action < SIZE * SIZE:
            raise ValueError(f"action {action} is outside the board")
        if self.is_terminal():
            raise ValueError("cannot step a terminal game")

        legal = self.legal_actions()
        if action not in legal:
            row, col = divmod(action, SIZE)
            if self.board.grid[action] != EMPTY:
                reason = "cell is already occupied"
            elif action in self.rules.forbidden_moves(self.board):
                reason = "move is forbidden by the current rules"
            else:
                reason = "move is not legal under the current rules"
            raise ValueError(f"illegal action {action} ({row}, {col}): {reason}")

        row, col = divmod(action, SIZE)
        self.board.play(row, col)

    def is_terminal(self) -> bool:
        return self.rules.is_terminal(self.board)

    def result(self) -> tuple[str, str] | None:
        if not self.is_terminal():
            return None
        if self.rules.is_win(self.board):
            if not self.board.history:
                raise ValueError("cannot determine winner without move history")
            winner = int(self.board.grid[self.board.history[-1]])
            winner_view = "black" if winner == BLACK else "white" if winner == WHITE else "none"
            return winner_view, "five_in_row"
        return "draw", "board_full"

    def render(self) -> str:
        return render_board(self.board)

    def clone(self) -> Game:
        clone = Game(self.rules)
        clone.board = self.board.copy()
        return clone
