from __future__ import annotations

import numpy as np

from gomoku.board import BLACK, SIZE, WHITE

N_PLANES = 4
N_ACTIONS = SIZE * SIZE
N_SYMMETRIES = 8

# Dihedral group of the square, applied to the last two axes.
_OPS = (
    lambda a: a,
    lambda a: np.rot90(a, 1, axes=(-2, -1)),
    lambda a: np.rot90(a, 2, axes=(-2, -1)),
    lambda a: np.rot90(a, 3, axes=(-2, -1)),
    lambda a: np.flip(a, axis=-2),
    lambda a: np.flip(a, axis=-1),
    lambda a: np.swapaxes(a, -1, -2),
    lambda a: np.flip(np.swapaxes(a, -1, -2), axis=(-2, -1)),
)


def board_to_planes(board, rules, player: int) -> np.ndarray:
    """Stack four planes for `player` to move: own stones, opponent stones, forbidden, colour."""
    planes = np.zeros((N_PLANES, SIZE, SIZE), dtype=np.uint8)
    planes[0] = (board.grid == player).reshape(SIZE, SIZE)
    opponent = WHITE if player == BLACK else BLACK
    planes[1] = (board.grid == opponent).reshape(SIZE, SIZE)
    forbidden = rules.forbidden_moves(board)
    if forbidden:
        planes[2].reshape(-1)[np.asarray(forbidden, dtype=np.int64)] = 1
    if player == BLACK:
        planes[3] = 1
    return planes


def symmetry_transform(array: np.ndarray, k: int) -> np.ndarray:
    """Apply symmetry k (0..7) to planes (..., 4, 11, 11) or a policy (..., 121).

    One function for both keeps features and policy under the same transform (hard rule 5).
    """
    if not 0 <= k < N_SYMMETRIES:
        raise ValueError(f"k must be in [0, {N_SYMMETRIES}), got {k}")
    arr = np.asarray(array)
    if arr.shape[-1] == N_ACTIONS:
        arr = arr.reshape(*arr.shape[:-1], SIZE, SIZE)
    out = _OPS[k](arr)
    return np.ascontiguousarray(out).reshape(array.shape)
