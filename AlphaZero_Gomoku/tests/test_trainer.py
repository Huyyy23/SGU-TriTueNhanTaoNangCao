import numpy as np
import pytest

from az.model import AZNet
from az.replay_buffer import ReplayBuffer
from az.trainer import Trainer
from config import DEFAULT_CONFIG
from gomoku.features import N_ACTIONS, N_PLANES, SIZE


def tiny_cfg(**overrides) -> dict:
    return {
        **DEFAULT_CONFIG,
        "n_blocks": 1,
        "n_filters": 8,
        "batch_size": 16,
        "train_steps_per_iter": 5,
        "n_iter": 2,
        "eval_fixed_batch_size": 32,
        "rule_set": "normal",
        **overrides,
    }


def filled_buffer(n: int = 64, seed: int = 0) -> ReplayBuffer:
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity=256)
    for _ in range(n):
        planes = rng.integers(0, 3, size=(N_PLANES, SIZE, SIZE), dtype=np.uint8)
        pi = np.zeros(N_ACTIONS, dtype=np.float32)
        pi[rng.integers(N_ACTIONS)] = 1.0
        buffer.add(planes, pi, float(rng.choice([-1.0, 1.0])), "normal")
    return buffer


def test_overfit_fixed_batch_drops_loss():
    # n_iter=100 keeps the cosine schedule from decaying to zero before the 300 steps finish.
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), tiny_cfg(n_iter=100))
    rng = np.random.default_rng(7)
    features = rng.integers(0, 3, size=(16, N_PLANES, SIZE, SIZE)).astype(np.float32)
    pi = np.zeros((16, N_ACTIONS), dtype=np.float32)
    pi[np.arange(16), rng.integers(0, N_ACTIONS, 16)] = 1.0
    z = np.array([1.0, -1.0] * 8, dtype=np.float32)

    first = trainer.train_on_batch(features, pi, z)["loss"]
    last = first
    for _ in range(300):
        last = trainer.train_on_batch(features, pi, z)["loss"]

    assert last < 0.5 * first, f"loss went {first:.4f} -> {last:.4f}"


def test_save_and_load_continue_from_the_same_step(tmp_path):
    cfg = tiny_cfg()
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer(), seed=1)
    trainer.freeze_eval_batch()
    trainer.train_steps(3)
    before = trainer.loss_on_fixed_batch()
    trainer.n_iter = 1
    trainer.save(tmp_path / "last.pt", tmp_path / "buffer_last.npz")

    restored_buffer = ReplayBuffer.load_npz(tmp_path / "buffer_last.npz")
    restored = Trainer.load(tmp_path / "last.pt", cfg, buffer=restored_buffer)

    assert restored.global_step == 3
    assert restored.n_iter == 1
    assert len(restored_buffer) == len(trainer.buffer)
    assert restored.eval_batch is not None
    assert np.array_equal(restored.eval_indices, trainer.eval_indices)
    assert restored.loss_on_fixed_batch() == pytest.approx(before, rel=1e-5)

    # The schedule resumes from step 3, not from step 0.
    restored.train_steps(1)
    lr = restored.optimizer.param_groups[0]["lr"]
    progress = 3 / restored.total_steps
    expected = cfg["learning_rate"] * 0.5 * (1.0 + np.cos(np.pi * progress))
    assert lr == pytest.approx(expected, rel=1e-6)


def test_train_steps_reports_distinct_samples_actually_used():
    cfg = tiny_cfg(batch_size=8, train_steps_per_iter=2)
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer(4))

    stats = trainer.train_steps()

    assert 1 <= stats["distinct_samples"] <= 4
    assert trainer.global_step == 2


def test_frozen_eval_samples_are_never_drawn_for_training():
    buffer = filled_buffer(32)
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), tiny_cfg(batch_size=8), buffer=buffer, seed=3)
    trainer.freeze_eval_batch(8)
    blocked = {int(i) for i in trainer.eval_indices}

    drawn: set[int] = set()
    rng = np.random.default_rng(0)
    for _ in range(40):
        _, _, _, indices = buffer.sample(
            8, "normal", rng, return_indices=True, exclude_indices=trainer.eval_indices
        )
        drawn.update(int(i) for i in indices)

    assert not (drawn & blocked)


def test_full_size_net_takes_one_training_step():
    cfg = tiny_cfg(n_blocks=6, n_filters=64)
    trainer = Trainer(AZNet(n_blocks=6, n_filters=64), cfg, buffer=filled_buffer(4))

    stats = trainer.train_steps(1)

    assert stats["loss"] > 0
    assert trainer.global_step == 1


def test_empty_buffer_refuses_to_train():
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), tiny_cfg(), buffer=ReplayBuffer(capacity=8))
    with pytest.raises(ValueError):
        trainer.train_steps(1)
