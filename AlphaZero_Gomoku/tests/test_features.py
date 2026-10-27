import numpy as np
import pytest

from gomoku.board import BLACK, SIZE, WHITE, Board
from gomoku.features import N_ACTIONS, N_PLANES, N_SYMMETRIES, board_to_planes, symmetry_transform
from gomoku.rules.normal import NormalRules


class ForbiddingRules(NormalRules):
    def forbidden_moves(self, board):
        return [0, 60]


def test_planes_shape_and_dtype():
    planes = board_to_planes(Board(), NormalRules(), BLACK)
    assert planes.shape == (N_PLANES, SIZE, SIZE)
    assert planes.dtype == np.uint8


def test_planes_contain_both_colours_and_colour_plane():
    board = Board()
    board.play(0, 0)  # Black
    board.play(5, 5)  # White
    planes = board_to_planes(board, NormalRules(), BLACK)
    assert planes[0, 0, 0] == 1
    assert planes[1, 5, 5] == 1
    assert planes[0].sum() == 1 and planes[1].sum() == 1
    assert planes[2].sum() == 0
    assert (planes[3] == 1).all()

    white_planes = board_to_planes(board, NormalRules(), WHITE)
    assert white_planes[0, 5, 5] == 1
    assert white_planes[1, 0, 0] == 1
    assert (white_planes[3] == 0).all()


def test_forbidden_plane_marks_forbidden_cells():
    planes = board_to_planes(Board(), ForbiddingRules(), BLACK)
    assert planes[2].reshape(-1)[0] == 1
    assert planes[2].reshape(-1)[60] == 1
    assert planes[2].sum() == 2


def test_symmetry_rejects_out_of_range_k():
    for k in (-1, N_SYMMETRIES):
        with pytest.raises(ValueError):
            symmetry_transform(np.zeros(N_ACTIONS), k)


def test_symmetry_gives_eight_distinct_variants():
    one_hot = np.zeros(N_ACTIONS, dtype=np.uint8)
    one_hot[1 * SIZE + 2] = 1

    seen = []
    for k in range(N_SYMMETRIES):
        assert np.count_nonzero(symmetry_transform(one_hot, k)) == 1
        seen.append(int(np.argmax(symmetry_transform(one_hot, k))))

    assert len(set(seen)) == N_SYMMETRIES
    assert seen == [1 * SIZE + 2, 8 * SIZE + 1, 9 * SIZE + 8, 2 * SIZE + 9,
                    9 * SIZE + 2, 1 * SIZE + 8, 2 * SIZE + 1, 8 * SIZE + 9]


def test_involutive_symmetries_are_their_own_inverse():
    rng = np.random.default_rng(0)
    array = rng.integers(0, 2, size=N_ACTIONS).astype(np.uint8)
    for k in (0, 2, 4, 5, 6, 7):
        once = symmetry_transform(array, k)
        assert np.array_equal(symmetry_transform(once, k), array)


def test_rotations_compose():
    rng = np.random.default_rng(1)
    array = rng.integers(0, 2, size=N_ACTIONS).astype(np.uint8)
    assert np.array_equal(symmetry_transform(symmetry_transform(array, 1), 3), array)
    assert np.array_equal(symmetry_transform(symmetry_transform(array, 3), 1), array)


def test_policy_and_planes_transform_consistently():
    board = Board()
    board.play(1, 2)  # Black stone at (1, 2)
    planes = board_to_planes(board, NormalRules(), BLACK)
    policy = np.zeros(N_ACTIONS, dtype=np.float32)
    policy[1 * SIZE + 2] = 1.0

    for k in range(N_SYMMETRIES):
        transformed_planes = symmetry_transform(planes, k)
        transformed_policy = symmetry_transform(policy, k)
        assert np.argmax(transformed_planes[0].reshape(-1)) == np.argmax(transformed_policy)
        assert np.sum(transformed_planes[0]) == 1
        assert np.isclose(transformed_policy.sum(), 1.0)


def test_symmetry_accepts_batches():
    rng = np.random.default_rng(2)
    policies = rng.random((3, N_ACTIONS)).astype(np.float32)
    planes = rng.integers(0, 3, size=(3, N_PLANES, SIZE, SIZE)).astype(np.uint8)

    transformed_policies = symmetry_transform(policies, 3)
    transformed_planes = symmetry_transform(planes, 3)
    assert transformed_policies.shape == policies.shape
    assert transformed_planes.shape == planes.shape
    for i in range(3):
        assert np.array_equal(transformed_policies[i], symmetry_transform(policies[i], 3))
        assert np.array_equal(transformed_planes[i], symmetry_transform(planes[i], 3))
