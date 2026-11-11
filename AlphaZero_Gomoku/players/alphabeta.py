"""Depth-limited minimax with optional alpha-beta pruning."""

from __future__ import annotations

import math

from gomoku.board import BLACK, EMPTY, SIZE, WHITE
from gomoku.patterns import find_patterns
from players.base import Player

_PATTERN_VALUE = {"open_four": 100_000, "blocked_four": 10_000, "open_three": 1_000, "blocked_three": 100, "broken_three": 100}
_WIN = 10**9


class AlphaBetaPlayer(Player):
    """Evaluate both sides by pattern score; search legal moves only."""

    name = "alphabeta"

    def __init__(self, rules, depth: int = 4, pruning: bool = True) -> None:
        if depth <= 0:
            raise ValueError("depth must be positive")
        self.rules = rules
        self.depth = depth
        self.pruning = pruning
        self.nodes_visited = 0

    @staticmethod
    def _score(board, player: int) -> int:
        occupied = [int(i) for i in range(SIZE * SIZE) if board.grid[i] != EMPTY]
        if not occupied:
            return 0
        candidates = set()
        for index in occupied:
            r, c = divmod(index, SIZE)
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < SIZE and 0 <= cc < SIZE and board.grid[rr * SIZE + cc] == EMPTY:
                        candidates.add(rr * SIZE + cc)
        score = 0
        for move in candidates:
            r, c = divmod(move, SIZE)
            score += sum(_PATTERN_VALUE.get(p.kind, 0) for p in find_patterns(board, r, c, player))
        return score

    def _evaluate(self, board, perspective: int) -> int:
        opponent = WHITE if perspective == BLACK else BLACK
        return self._score(board, perspective) - self._score(board, opponent)

    def _search(self, board, depth: int, perspective: int, alpha: float, beta: float) -> int:
        self.nodes_visited += 1
        if self.rules.is_win(board):
            winner = WHITE if board.current_player == BLACK else BLACK
            return _WIN + depth if winner == perspective else -_WIN - depth
        if depth == 0 or self.rules.is_terminal(board):
            return self._evaluate(board, perspective)

        moves = self.rules.legal_moves(board)
        moves.sort(key=lambda m: self._score_after(board, m, board.current_player), reverse=True)
        maximizing = board.current_player == perspective
        best = -math.inf if maximizing else math.inf
        for move in moves:
            board.play(*divmod(move, SIZE))
            value = self._search(board, depth - 1, perspective, alpha, beta)
            board.undo()
            if maximizing:
                best = max(best, value)
                alpha = max(alpha, best)
            else:
                best = min(best, value)
                beta = min(beta, best)
            if self.pruning and alpha >= beta:
                break
        return int(best)

    def _score_after(self, board, move: int, player: int) -> int:
        r, c = divmod(move, SIZE)
        score = sum(_PATTERN_VALUE.get(p.kind, 0) for p in find_patterns(board, r, c, player))
        opponent = WHITE if player == BLACK else BLACK
        return score + self._score(board, opponent) // 10

    def select_move(self, board) -> tuple[int, int]:
        legal = self.rules.legal_moves(board)
        if not legal or self.rules.is_terminal(board):
            raise RuntimeError("AlphaBetaPlayer asked to move on a terminal board")
        self.nodes_visited = 0
        player = board.current_player
        ordered = sorted(legal, key=lambda m: self._score_after(board, m, player), reverse=True)
        best_move, best_value = ordered[0], -math.inf
        alpha, beta = -math.inf, math.inf
        for move in ordered:
            board.play(*divmod(move, SIZE))
            value = self._search(board, self.depth - 1, player, alpha, beta)
            board.undo()
            if value > best_value:
                best_move, best_value = move, value
            if self.pruning:
                alpha = max(alpha, best_value)
        return divmod(best_move, SIZE)
