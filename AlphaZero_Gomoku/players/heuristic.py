"""Pattern-scored Gomoku baseline; no neural network required."""

from __future__ import annotations

from gomoku.board import BLACK, EMPTY, SIZE, WHITE
from gomoku.patterns import find_patterns
from players.base import Player

_PATTERN_VALUE = {
    "open_four": 100_000,
    "blocked_four": 10_000,
    "open_three": 1_000,
    "blocked_three": 100,
    "broken_three": 100,
}


class HeuristicPlayer(Player):
    """Priority: win now > block win now > make open four > block open four > make open three > position.

    Within one priority, prefer positional score, then lowest flat cell index.
    """

    name = "heuristic"

    def __init__(self, rules) -> None:
        self.rules = rules

    @staticmethod
    def _pattern_score(board, move: int, player: int) -> int:
        r, c = divmod(move, SIZE)
        return sum(_PATTERN_VALUE.get(pattern.kind, 0) for pattern in find_patterns(board, r, c, player))

    @staticmethod
    def _position_score(move: int) -> int:
        r, c = divmod(move, SIZE)
        return 2 * SIZE - abs(2 * r - (SIZE - 1)) - abs(2 * c - (SIZE - 1))

    def select_move(self, board) -> tuple[int, int]:
        legal = self.rules.legal_moves(board)
        if not legal or self.rules.is_terminal(board):
            raise RuntimeError("HeuristicPlayer asked to move on a terminal board")

        player = board.current_player
        opponent = WHITE if player == BLACK else BLACK
        candidates = []
        for move in legal:
            r, c = divmod(move, SIZE)
            own_score = self._pattern_score(board, move, player)
            board.play(r, c)
            wins = self.rules.is_win(board)
            board.undo()
            if wins:
                return r, c

            board.current_player = opponent
            opponent_legal = self.rules.legal_moves(board)
            if move in opponent_legal:
                board.play(r, c)
                opponent_wins = self.rules.is_win(board)
                board.undo()
            board.current_player = player
            opponent_score = self._pattern_score(board, move, opponent) if move in opponent_legal else 0

            priority = 4 if opponent_wins else 3 if own_score >= _PATTERN_VALUE["open_four"] else 2 if opponent_score >= _PATTERN_VALUE["open_four"] else 1 if own_score >= _PATTERN_VALUE["open_three"] else 0
            candidates.append((priority, self._position_score(move), -move, move))

        return divmod(max(candidates)[-1], SIZE)
