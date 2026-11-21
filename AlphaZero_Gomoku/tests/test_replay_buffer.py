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


def test_resume_full_buffer_preserves_oldest_sample_order(tmp_path):
    buffer = ReplayBuffer(capacity=2)
    for col in range(3):
        features, pi = make_sample(1, col)
        buffer.add(features, pi, 1.0, "normal")
    path = tmp_path / "buffer.npz"
    buffer.save_npz(path)
    resumed = ReplayBuffer.load_npz(path)

    features, pi = make_sample(1, 3)
    buffer.add(features, pi, -1.0, "normal")
    resumed.add(features, pi, -1.0, "normal")

    np.testing.assert_array_equal(resumed.features, buffer.features)
    np.testing.assert_array_equal(resumed.pi, buffer.pi)
    np.testing.assert_array_equal(resumed.z, buffer.z)


def test_failed_replay_save_preserves_previous_file(tmp_path, monkeypatch):
    buffer = ReplayBuffer(capacity=2)
    features, pi = make_sample()
    buffer.add(features, pi, 1.0, "normal")
    path = tmp_path / "buffer.npz"
    buffer.save_npz(path)
    original = path.read_bytes()

    def interrupted_save(target, **arrays):
        with open(target, "wb") as stream:
            stream.write(b"incomplete")
        raise OSError("interrupted write")

    monkeypatch.setattr(np, "savez", interrupted_save)
    with pytest.raises(OSError, match="interrupted"):
        buffer.save_npz(path)
    assert path.read_bytes() == original


def test_loading_legacy_full_buffer_warns_age_order_unknown(tmp_path):
    buffer = ReplayBuffer(capacity=2)
    features, pi = make_sample()
    buffer.add(features, pi, 1.0, "normal")
    buffer.add(features, pi, -1.0, "normal")
    path = tmp_path / "legacy.npz"
    np.savez(path, features=buffer.features, pi=buffer.pi, z=buffer.z,
             rule_set=buffer.rule_set, capacity=np.array(buffer.capacity))

    with pytest.warns(UserWarning, match="oldest"):
        loaded = ReplayBuffer.load_npz(path)
    np.testing.assert_array_equal(loaded.features, buffer.features)


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


def test_game_lineage_survives_ring_save_load_and_symmetry(tmp_path):
    buffer = ReplayBuffer(capacity=2)
    features, pi = make_sample()
    for game_id in ("old", "game-a", "game-b"):
        buffer.add(features, pi, 1.0, "normal", game_id=game_id)
    path = tmp_path / "replay.npz"
    buffer.save_npz(path)
    restored = ReplayBuffer.load_npz(path)
    assert set(restored.game_ids[:len(restored)]) == {"game-a", "game-b"}
    _, _, _, indices = restored.sample(20, "normal", np.random.default_rng(7), return_indices=True)
    assert set(restored.game_ids[indices]) == {"game-a", "game-b"}
    restored.add(features, pi, 0.0, "normal", game_id="game-c")
    assert set(restored.game_ids[:len(restored)]) == {"game-b", "game-c"}


def test_legacy_replay_has_unknown_game_lineage(tmp_path):
    features, pi = make_sample()
    path = tmp_path / "legacy.npz"
    np.savez(path, features=features[None], pi=pi[None], z=np.array([1.0]),
             rule_set=np.array([0], dtype=np.uint8), capacity=np.array(2))
    restored = ReplayBuffer.load_npz(path)
    assert restored.game_ids[0] == ""
