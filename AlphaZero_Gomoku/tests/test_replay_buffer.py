import numpy as np
import pytest
import torch

from az.replay_buffer import RULE_SET_IDS, ReplayBuffer
from gomoku.board import SIZE
from gomoku.features import N_ACTIONS, N_PLANES


def make_sample(row: int = 1, col: int = 2) -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((N_PLANES, SIZE, SIZE), dtype=np.uint8)
    features[0, row, col] = 1
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[row * SIZE + col] = 1.0
    return features, pi


def test_add_stores_sample_with_rule_set_label():
    buffer = ReplayBuffer(capacity=10)
    features, pi = make_sample()
    buffer.add(features, pi, 1.0, "normal")

    assert len(buffer) == 1
    assert buffer.pi.dtype == np.float16
    assert buffer.features.dtype == np.uint8
    assert buffer.z[0] == 1.0
    assert buffer.rule_set[0] == RULE_SET_IDS["normal"]


def test_sample_returns_float32_torch_batch():
    buffer = ReplayBuffer(capacity=10)
    for i in range(5):
        features, pi = make_sample()
        buffer.add(features, pi, float(i), "normal")

    features, pi, z = buffer.sample(8, "normal", rng=np.random.default_rng(0))

    assert isinstance(features, torch.Tensor) and features.dtype == torch.float32
    assert features.shape == (8, N_PLANES, SIZE, SIZE)
    assert pi.shape == (8, N_ACTIONS)
    assert z.shape == (8,)
    assert torch.allclose(pi.sum(dim=1), torch.ones(8))


def test_mixing_rule_sets_raises():
    buffer = ReplayBuffer(capacity=10)
    features, pi = make_sample()
    buffer.add(features, pi, 0.0, "normal")
    with pytest.raises(ValueError):
        buffer.add(features, pi, 0.0, "renju")


def test_unknown_rule_set_raises():
    buffer = ReplayBuffer(capacity=10)
    features, pi = make_sample()
    with pytest.raises(ValueError):
        buffer.add(features, pi, 0.0, "gomoku")
    buffer.add(features, pi, 0.0, "normal")
    with pytest.raises(ValueError):
        buffer.sample(1, "gomoku")


def test_sampling_with_wrong_rule_set_raises():
    buffer = ReplayBuffer(capacity=10)
    features, pi = make_sample()
    buffer.add(features, pi, 0.0, "normal")
    with pytest.raises(ValueError):
        buffer.sample(1, "renju")


def test_capacity_overwrites_oldest():
    buffer = ReplayBuffer(capacity=2)
    features, pi = make_sample()
    buffer.add(features, pi, 1.0, "normal")
    buffer.add(features, pi, 2.0, "normal")
    buffer.add(features, pi, 3.0, "normal")

    assert len(buffer) == 2
    assert sorted(buffer.z[:2].tolist()) == [2.0, 3.0]


def test_empty_buffer_cannot_sample():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=4).sample(1, "normal")


def test_invalid_shapes_raise():
    buffer = ReplayBuffer(capacity=4)
    features, pi = make_sample()
    with pytest.raises(ValueError):
        buffer.add(features[:, :5, :5], pi, 0.0, "normal")
    with pytest.raises(ValueError):
        buffer.add(features, pi[:10], 0.0, "normal")


def test_augmentation_keeps_policy_and_features_in_the_same_cell():
    buffer = ReplayBuffer(capacity=4)
    features, pi = make_sample(1, 2)
    buffer.add(features, pi, -1.0, "normal")

    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(80):
        features, pi, z = buffer.sample(1, "normal", rng=rng)
        feature_cell = int(features[0, 0].reshape(-1).argmax())
        policy_cell = int(pi[0].argmax())
        assert feature_cell == policy_cell
        assert torch.allclose(pi.sum(dim=1), torch.ones(1))
        assert z[0].item() == -1.0
        seen.add(feature_cell)

    assert len(seen) == 8
