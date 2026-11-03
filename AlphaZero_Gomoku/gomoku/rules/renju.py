"""Renju rules for Black: double-three, double-four and overline are forbidden."""

from __future__ import annotations

import numpy as np

from gomoku.board import BLACK, EMPTY, SIZE, WHITE
from gomoku.rules.base import Rules

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class RenjuRules(Rules):
    """Black wins with exactly five; White wins with five or more."""

    name = "renju"

    def __init__(self) -> None:
        self._cache_key: tuple[bytes, int] | None = None
        self._cache_value: tuple[int, ...] = ()

    def legal_moves(self, board) -> list[int]:
        empty = board.grid == EMPTY
        if board.current_player != BLACK:
            return np.flatnonzero(empty).tolist()

        forbidden = self.forbidden_moves(board)
        if forbidden:
            empty[np.asarray(forbidden, dtype=np.intp)] = False
        return np.flatnonzero(empty).tolist()

    def forbidden_moves(self, board) -> list[int]:
        if board.current_player != BLACK:
            return []

        key = (board.grid.tobytes(), board.current_player)
        if key == self._cache_key:
            return list(self._cache_value)

        grid = board.grid.reshape(SIZE, SIZE)
        candidates = self._candidate_mask(grid)
        memo: dict[tuple[bytes, int, int], bool] = {}
        forbidden: list[int] = []

        for move in np.flatnonzero(candidates & (grid == EMPTY)):
            r, c = divmod(int(move), SIZE)
            grid[r, c] = BLACK
            try:
                is_forbidden = self._is_forbidden_after(grid, r, c, memo)
            finally:
                grid[r, c] = EMPTY
            if is_forbidden:
                forbidden.append(int(move))

        self._cache_key = key
        self._cache_value = tuple(forbidden)
        return forbidden

    def is_win(self, board) -> bool:
        grid = board.grid.reshape(SIZE, SIZE)
        if board.history:
            r, c = divmod(board.history[-1], SIZE)
            player = int(grid[r, c])
            if player == BLACK:
                return self._has_exact_five(grid, r, c)
            if player == WHITE:
                return self._has_five_or_more(grid, r, c)
            return False

        return self._has_any_winning_line(grid)

    def is_terminal(self, board) -> bool:
        return self.is_win(board) or not self.legal_moves(board)

    @staticmethod
    def _candidate_mask(grid: np.ndarray) -> np.ndarray:
        """Mark cells within four line steps of an existing Black stone."""
        black = grid == BLACK
        candidates = np.zeros((SIZE, SIZE), dtype=bool)
        for dr, dc in _DIRECTIONS:
            for distance in range(-4, 5):
                if distance == 0:
                    continue
                shift_r, shift_c = dr * distance, dc * distance
                src_r = slice(max(0, -shift_r), min(SIZE, SIZE - shift_r))
                src_c = slice(max(0, -shift_c), min(SIZE, SIZE - shift_c))
                dst_r = slice(max(0, shift_r), min(SIZE, SIZE + shift_r))
                dst_c = slice(max(0, shift_c), min(SIZE, SIZE + shift_c))
                candidates[dst_r, dst_c] |= black[src_r, src_c]
        return candidates

    @staticmethod
    def _run_length(
        grid: np.ndarray, r: int, c: int, dr: int, dc: int, player: int
    ) -> int:
        count = 1
        for sign in (-1, 1):
            rr, cc = r + sign * dr, c + sign * dc
            while 0 <= rr < SIZE and 0 <= cc < SIZE and grid[rr, cc] == player:
                count += 1
                rr += sign * dr
                cc += sign * dc
        return count

    @classmethod
    def _has_exact_five(cls, grid: np.ndarray, r: int, c: int) -> bool:
        return any(
            cls._run_length(grid, r, c, dr, dc, BLACK) == 5
            for dr, dc in _DIRECTIONS
        )

    @classmethod
    def _has_five_or_more(cls, grid: np.ndarray, r: int, c: int) -> bool:
        return any(
            cls._run_length(grid, r, c, dr, dc, WHITE) >= 5
            for dr, dc in _DIRECTIONS
        )

    @classmethod
    def _has_any_winning_line(cls, grid: np.ndarray) -> bool:
        for r, c in np.argwhere(grid != EMPTY):
            player = int(grid[r, c])
            if player == BLACK and cls._has_exact_five(grid, int(r), int(c)):
                return True
            if player == WHITE and cls._has_five_or_more(grid, int(r), int(c)):
                return True
        return False

    def _is_forbidden_after(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        memo: dict[tuple[bytes, int, int], bool],
    ) -> bool:
        key = (grid.tobytes(), r, c)
        if key in memo:
            return memo[key]

        # An exact five is legal even if the same move creates other threats.
        if self._has_exact_five(grid, r, c):
            memo[key] = False
            return False
        if any(
            self._run_length(grid, r, c, dr, dc, BLACK) > 5
            for dr, dc in _DIRECTIONS
        ):
            memo[key] = True
            return True
        if len(self._four_groups(grid, r, c)) >= 2:
            memo[key] = True
            return True

        result = len(self._open_three_groups(grid, r, c, memo)) >= 2
        memo[key] = result
        return result

    def _four_groups(
        self, grid: np.ndarray, r: int, c: int
    ) -> set[tuple[int, int, frozenset[tuple[int, int]]]]:
        groups: set[tuple[int, int, frozenset[tuple[int, int]]]] = set()
        for dr, dc in _DIRECTIONS:
            for start in range(-4, 1):
                offsets = range(start, start + 5)
                cells = [(r + i * dr, c + i * dc) for i in offsets]
                if any(
                    not (0 <= rr < SIZE and 0 <= cc < SIZE)
                    for rr, cc in cells
                ):
                    continue
                values = [int(grid[rr, cc]) for rr, cc in cells]
                if values.count(BLACK) != 4 or values.count(EMPTY) != 1:
                    continue

                empty_r, empty_c = cells[values.index(EMPTY)]
                grid[empty_r, empty_c] = BLACK
                try:
                    exact_five = (
                        self._run_length(
                            grid, empty_r, empty_c, dr, dc, BLACK
                        )
                        == 5
                    )
                finally:
                    grid[empty_r, empty_c] = EMPTY
                if exact_five:
                    stones = frozenset(
                        (rr, cc)
                        for (rr, cc), value in zip(cells, values)
                        if value == BLACK
                    )
                    groups.add((dr, dc, stones))
        return groups

    def _open_three_groups(
        self,
        grid: np.ndarray,
        r: int,
        c: int,
        memo: dict[tuple[bytes, int, int], bool],
    ) -> set[tuple[int, int, frozenset[tuple[int, int]]]]:
        threes: set[tuple[int, int, frozenset[tuple[int, int]]]] = set()
        for dr, dc in _DIRECTIONS:
            for offset in range(-4, 5):
                if offset == 0:
                    continue
                extension_r = r + offset * dr
                extension_c = c + offset * dc
                if not (
                    0 <= extension_r < SIZE
                    and 0 <= extension_c < SIZE
                    and grid[extension_r, extension_c] == EMPTY
                ):
                    continue

                extended = grid.copy()
                extended[extension_r, extension_c] = BLACK
                open_fours = self._open_four_groups(
                    extended, r, c, dr, dc
                )
                if not open_fours or self._is_forbidden_after(
                    extended, extension_r, extension_c, memo
                ):
                    continue
                for stones in open_fours:
                    original_stones = stones - {(extension_r, extension_c)}
                    if len(original_stones) == 3 and (r, c) in original_stones:
                        threes.add((dr, dc, frozenset(original_stones)))
        return threes

    @staticmethod
    def _open_four_groups(
        grid: np.ndarray, r: int, c: int, dr: int, dc: int
    ) -> set[frozenset[tuple[int, int]]]:
        groups: set[frozenset[tuple[int, int]]] = set()
        for start in range(-5, 1):
            offsets = range(start, start + 6)
            cells = [(r + offset * dr, c + offset * dc) for offset in offsets]
            if any(
                not (0 <= rr < SIZE and 0 <= cc < SIZE)
                for rr, cc in cells
            ):
                continue
            if (
                grid[cells[0]] != EMPTY
                or grid[cells[-1]] != EMPTY
                or any(grid[rr, cc] != BLACK for rr, cc in cells[1:-1])
                or (r, c) not in cells[1:-1]
            ):
                continue
            groups.add(frozenset(cells[1:-1]))
        return groups
