from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

_native = None
_active_backend = "python"


def set_backend(backend: str) -> str:
    """Select Python or native pattern detection; explicit native never falls back."""
    global _native, _active_backend
    if backend not in ("python", "cpp", "native"):
        raise ValueError("patterns backend must be python or cpp")
    if backend == "python":
        _native = None
        _active_backend = "python"
        return _active_backend
    try:
        import gomoku_native
    except (ImportError, OSError) as exc:
        raise ImportError("GOMOKU_PATTERN_BACKEND=native requires the gomoku_native extension") from exc
    _native = gomoku_native
    _active_backend = "cpp"
    return _active_backend


_requested_backend = os.environ.get("GOMOKU_PATTERN_BACKEND", "python").lower()
if _requested_backend not in ("auto", "python", "native", "cpp"):
    raise ValueError("GOMOKU_PATTERN_BACKEND must be python, native, or auto")
if _requested_backend in ("native", "cpp"):
    set_backend("cpp")
elif _requested_backend == "auto":
    # Import-time auto-loading bypasses the process-level compiler and freshness checks.
    pass

@dataclass(frozen=True)
class Pattern:
    kind: str
    direction: tuple[int, int]
    completion_cells: tuple[tuple[int, int], ...]


def find_patterns(board: Board, r: int, c: int, player: int) -> list[Pattern]:
    """Classify patterns formed by hypothetically placing a stone."""
    if not isinstance(board, Board):
        raise ValueError("board must be a Board")
    if not (0 <= r < SIZE and 0 <= c < SIZE):
        raise ValueError(f"Cell ({r}, {c}) is outside the board.")
    if player not in (BLACK, WHITE):
        raise ValueError(f"Invalid player: {player}")
    if board.at(r, c) != EMPTY:
        raise ValueError(f"Cell ({r}, {c}) is already occupied.")

    if _native is not None:
        grid = np.ascontiguousarray(board.as_array(), dtype=np.int8)
        return [Pattern(*pattern) for pattern in _native.find_patterns(grid, r, c, int(player))]

    grid = board.as_array().tolist()
    patterns = []
    for dr, dc in _DIRECTIONS:
        cells = [(r, c)]
        for sign in (-1, 1):
            rr, cc = r + sign * dr, c + sign * dc
            while 0 <= rr < SIZE and 0 <= cc < SIZE and grid[rr][cc] == player:
                cells.append((rr, cc))
                rr += sign * dr
                cc += sign * dc
        cells.sort(key=lambda cell: cell[0] * dr + cell[1] * dc)
        length = len(cells)
        ends = []
        for rr, cc in ((cells[0][0] - dr, cells[0][1] - dc), (cells[-1][0] + dr, cells[-1][1] + dc)):
            if 0 <= rr < SIZE and 0 <= cc < SIZE and grid[rr][cc] == EMPTY:
                ends.append((rr, cc))

        if length > 5:
            patterns.append(Pattern("overline", (dr, dc), ()))
        elif length == 4:
            patterns.append(Pattern("open_four" if len(ends) == 2 else "blocked_four", (dr, dc), tuple(ends)))
        elif length == 3:
            patterns.append(Pattern("open_three" if len(ends) == 2 else "blocked_three", (dr, dc), tuple(ends)))

        if length <= 3:
            gap_cells = set()
            for start in range(-4, 1):
                end = start + 4
                if not (0 <= r + start * dr < SIZE and 0 <= c + start * dc < SIZE
                        and 0 <= r + end * dr < SIZE and 0 <= c + end * dc < SIZE):
                    continue
                window = [(r + (start + i) * dr, c + (start + i) * dc) for i in range(5)]
                values = [player if i == -start else grid[cell[0]][cell[1]] for i, cell in enumerate(window)]
                if values.count(player) != 3 or values.count(EMPTY) != 2 or any(v not in (EMPTY, player) for v in values):
                    continue
                stones = [i for i, value in enumerate(values) if value == player]
                if stones[-1] - stones[0] > 2:
                    gap_cells.update(window[i] for i in range(stones[0] + 1, stones[-1]) if values[i] == EMPTY)
            if gap_cells:
                patterns.append(Pattern("broken_three", (dr, dc), tuple(sorted(gap_cells))))
    return patterns
