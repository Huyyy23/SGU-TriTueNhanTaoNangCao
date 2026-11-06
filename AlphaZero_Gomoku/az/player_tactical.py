from __future__ import annotations

from gomoku.board import SIZE


class TacticalAZPlayer:
    """AZ player with a shallow tactical override (win now, block opponent win)."""

    def __init__(self, base_player) -> None:
        self.base = base_player

    def select_move(self, board) -> tuple[int, int]:
        rules = self.base.rules
        legal = set(rules.legal_moves(board))

        # 1. Win immediately if possible.
        for move in legal:
            r, c = divmod(move, SIZE)
            board.play(r, c)
            is_win = rules.is_win(board)
            board.undo()
            if is_win:
                return (r, c)

        # 2. Block opponent's immediate win.
        opponent = 3 - board.current_player
        for move in legal:
            r, c = divmod(move, SIZE)
            board.play(r, c)  # temporarily as current player? need opponent perspective
            board.undo()
        # Re-evaluate blocking by simulating opponent turn.
        for move in legal:
            r, c = divmod(move, SIZE)
            # place opponent stone
            saved = board.current_player
            board.current_player = opponent
            board.play(r, c)
            is_win = rules.is_win(board)
            board.undo()
            board.current_player = saved
            if is_win:
                return (r, c)

        return self.base.select_move(board)