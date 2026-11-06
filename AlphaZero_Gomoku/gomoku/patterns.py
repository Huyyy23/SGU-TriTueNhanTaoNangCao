from __future__ import annotations

from dataclasses import dataclass

from gomoku.board import BLACK, EMPTY, SIZE, WHITE, Board

_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True)
class Pattern:
    kind: str
    direction: tuple[int, int]
    completion_cells: tuple[tuple[int, int], ...]


def find_patterns(board: Board, r: int, c: int, player: int) -> list[Pattern]:
    """Classify contiguous patterns formed by hypothetically placing a stone."""
    if not isinstance(board, Board):
        raise ValueError("board must be a Board")
    if not (0 <= r < SIZE and 0 <= c < SIZE):
        raise ValueError(f"Cell ({r}, {c}) is outside the board.")
    if player not in (BLACK, WHITE):
        raise ValueError(f"Invalid player: {player}")
    if board.at(r, c) != EMPTY:
        raise ValueError(f"Cell ({r}, {c}) is already occupied.")

    grid = board.as_array()
    patterns = []
    for dr, dc in _DIRECTIONS:
        cells = [(r, c)]
        for sign in (-1, 1):
            rr, cc = r + sign * dr, c + sign * dc
            while 0 <= rr < SIZE and 0 <= cc < SIZE and int(grid[rr, cc]) == player:
                cells.append((rr, cc))
                rr += sign * dr
                cc += sign * dc
        cells.sort(key=lambda cell: cell[0] * dr + cell[1] * dc)
        length = len(cells)
        ends = []
        for rr, cc in ((cells[0][0] - dr, cells[0][1] - dc),
                       (cells[-1][0] + dr, cells[-1][1] + dc)):
            if 0 <= rr < SIZE and 0 <= cc < SIZE and int(grid[rr, cc]) == EMPTY:
                ends.append((rr, cc))

        if length > 5:
            patterns.append(Pattern("overline", (dr, dc), ()))
        elif length == 4:
            patterns.append(Pattern("open_four" if len(ends) == 2 else "blocked_four", (dr, dc), tuple(ends)))
        elif length == 3 and len(ends) == 2:
            patterns.append(Pattern("open_three", (dr, dc), tuple(ends)))
    return patterns
