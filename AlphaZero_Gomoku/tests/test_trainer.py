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
    trainer.promotion_eval_games = 123
    trainer.rng.random(5)
    pointer = tmp_path / "latest.json"
    trainer.save(pointer)
    next_rng_value = trainer.rng.random()

    from az.trainer import resolve_generation
    checkpoint, replay = resolve_generation(pointer)
    restored_buffer = ReplayBuffer.load_npz(replay)
    restored = Trainer.load(checkpoint, cfg, buffer=restored_buffer)

    assert restored.global_step == 3
    assert restored.n_iter == 1
    assert restored.promotion_eval_games == 123
    assert restored.rng.random() == next_rng_value
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


def test_resume_rejects_replay_from_another_checkpoint(tmp_path):
    cfg = tiny_cfg()
    first = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer(seed=1))
    second = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer(seed=2))
    from az.trainer import resolve_generation
    first.save(tmp_path / "first.json")
    second.save(tmp_path / "second.json")
    first_checkpoint, _ = resolve_generation(tmp_path / "first.json")
    _, second_replay = resolve_generation(tmp_path / "second.json")
    wrong_buffer = ReplayBuffer.load_npz(second_replay)
    with pytest.raises(ValueError, match="replay.*checkpoint|checkpoint.*replay"):
        Trainer.load(first_checkpoint, cfg, buffer=wrong_buffer)


def test_generation_commit_crashes_and_explicit_rollback(tmp_path, monkeypatch):
    import az.trainer as persistence

    pointer = tmp_path / "latest.json"
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), tiny_cfg(), buffer=filled_buffer())
    first = trainer.save(pointer)
    first_id = first.parent.name
    original = first.read_bytes()
    real_write = persistence._write_pointer

    def crash_before(*args):
        raise RuntimeError("before commit")

    monkeypatch.setattr(persistence, "_write_pointer", crash_before)
    with pytest.raises(RuntimeError, match="before commit"):
        trainer.save(pointer)
    orphan = next(path for path in (tmp_path / "generations").iterdir() if path != first.parent)
    with pytest.raises(ValueError, match="committed history"):
        persistence.resolve_generation(pointer, orphan.name)
    assert persistence.resolve_generation(pointer)[0] == first

    def crash_after(*args):
        real_write(*args)
        raise RuntimeError("after commit")

    monkeypatch.setattr(persistence, "_write_pointer", crash_after)
    with pytest.raises(RuntimeError, match="after commit"):
        trainer.save(pointer)
    second, _ = persistence.resolve_generation(pointer)
    assert second != first
    assert persistence.resolve_generation(pointer, first_id)[0] == first
    assert first.read_bytes() == original


def test_rollback_save_records_actual_parent_and_preserves_abandoned_generation(tmp_path):
    import json
    from az.trainer import resolve_generation

    cfg = tiny_cfg()
    pointer = tmp_path / "latest.json"
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer())
    first = trainer.save(pointer)
    second = trainer.save(pointer)
    checkpoint, replay = resolve_generation(pointer, first.parent.name)
    rolled_back = Trainer.load(checkpoint, cfg, ReplayBuffer.load_npz(replay), device="cpu")
    branch = rolled_back.save(pointer)
    manifest = json.loads((branch.parent / "manifest.json").read_text())
    assert manifest["previous"] == first.parent.name
    assert resolve_generation(pointer, second.parent.name)[0] == second
    assert resolve_generation(pointer, first.parent.name)[0] == first


def test_generation_survives_process_termination_at_commit(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    import time

    from az.trainer import resolve_generation

    pointer = tmp_path / "latest.json"
    legacy = [tmp_path / "latest.pt", tmp_path / "buffer_latest.npz"]
    for path in legacy:
        path.write_bytes(b"historical artifact: " + path.name.encode())
    legacy_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in legacy]
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), tiny_cfg(), buffer=filled_buffer())
    trainer.n_iter = 1
    first = trainer.save(pointer)
    first_id = first.parent.name
    first_hashes = [hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in resolve_generation(pointer)]

    child = """
import sys, time
from pathlib import Path
import az.trainer as persistence
from az.replay_buffer import ReplayBuffer
from config import DEFAULT_CONFIG

pointer, marker, seam = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cfg = {**DEFAULT_CONFIG, "n_blocks": 1, "n_filters": 8, "batch_size": 16,
       "train_steps_per_iter": 5, "n_iter": 2, "eval_fixed_batch_size": 32,
       "rule_set": "normal"}
checkpoint, replay = persistence.resolve_generation(pointer)
trainer = persistence.Trainer.load(checkpoint, cfg, buffer=ReplayBuffer.load_npz(replay), device="cpu")
trainer.n_iter = 2
write_pointer = persistence._write_pointer
def stop_at_seam(*args):
    if seam == "after":
        write_pointer(*args)
    marker.write_text("ready", encoding="utf-8")
    while True:
        time.sleep(0.1)
persistence._write_pointer = stop_at_seam
trainer.save(pointer)
"""
    for seam in ("before", "after"):
        marker = tmp_path / (seam + ".ready")
        process = subprocess.Popen([sys.executable, "-c", child, str(pointer), str(marker), seam])
        try:
            deadline = time.monotonic() + 60
            while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert marker.exists(), f"child did not reach {seam} commit seam (exit={process.poll()})"
        finally:
            process.kill()
            process.wait(timeout=10)
        assert process.returncode != 0
        checkpoint, replay = resolve_generation(pointer)
        restored = Trainer.load(checkpoint, tiny_cfg(), buffer=ReplayBuffer.load_npz(replay), device="cpu")
        assert restored.n_iter == (1 if seam == "before" else 2)
        if seam == "before":
            assert checkpoint == first
            orphan = next(path for path in (tmp_path / "generations").iterdir() if path != first.parent)
            with pytest.raises(ValueError, match="committed history"):
                resolve_generation(pointer, orphan.name)
        else:
            assert checkpoint != first
            assert json.loads((checkpoint.parent / "manifest.json").read_text())["previous"] == first_id
            assert resolve_generation(pointer, first_id)[0] == first
        assert [hashlib.sha256(path.read_bytes()).hexdigest()
                for path in resolve_generation(pointer, first_id)] == first_hashes
        assert [hashlib.sha256(path.read_bytes()).hexdigest() for path in legacy] == legacy_hashes


def test_generation_rejects_tampering_and_traversal(tmp_path):
    from az.trainer import resolve_generation

    pointer = tmp_path / "latest.json"
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), tiny_cfg(), buffer=filled_buffer())
    checkpoint = trainer.save(pointer)
    with pytest.raises(ValueError, match="generation ID"):
        resolve_generation(pointer, "../outside")
    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_generation(pointer)


def test_resume_rejects_legacy_checkpoint_without_replay_identity(tmp_path):
    from az.model import save_checkpoint

    cfg = tiny_cfg()
    checkpoint = tmp_path / "legacy.pt"
    replay = tmp_path / "legacy.npz"
    save_checkpoint(AZNet(n_blocks=1, n_filters=8), checkpoint, meta={"rule_set": "normal"})
    filled_buffer().save_npz(replay)

    with pytest.raises(ValueError, match="replay.*checkpoint|checkpoint.*replay"):
        Trainer.load(checkpoint, cfg, buffer=ReplayBuffer.load_npz(replay))


def test_resume_rejects_full_legacy_replay_even_when_hash_matches(tmp_path):
    import hashlib
    from az.model import save_checkpoint

    cfg = tiny_cfg()
    buffer = filled_buffer(256)
    replay = tmp_path / "legacy.npz"
    np.savez(replay, features=buffer.features, pi=buffer.pi, z=buffer.z,
             rule_set=buffer.rule_set, capacity=np.array(buffer.capacity))
    checkpoint = tmp_path / "bound.pt"
    save_checkpoint(AZNet(n_blocks=1, n_filters=8), checkpoint,
                    meta={"rule_set": "normal", "replay_sha256": hashlib.sha256(replay.read_bytes()).hexdigest()})

    with pytest.warns(UserWarning, match="oldest"):
        restored = ReplayBuffer.load_npz(replay)
    with pytest.raises(ValueError, match="cursor"):
        Trainer.load(checkpoint, cfg, buffer=restored)


def test_train_steps_reports_distinct_samples_actually_used():
    cfg = tiny_cfg(batch_size=8, train_steps_per_iter=2)
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer(4))

    stats = trainer.train_steps()

    assert 1 <= stats["distinct_samples"] <= 4
    assert trainer.global_step == 2


def test_training_counts_only_identified_games_actually_sampled(monkeypatch):
    buffer = filled_buffer(4)
    buffer.game_ids[:4] = ["a" * 64, "a" * 64, "b" * 64, ""]
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), tiny_cfg(batch_size=32), buffer=buffer)
    monkeypatch.setattr(trainer, "train_on_batch", lambda *args: {
        "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0})
    stats = trainer.train_steps(1)
    assert stats["identified_games_used"] == 2
    assert stats["unidentified_samples_used"] == 1


def test_resume_rejects_missing_optimizer_or_rng(tmp_path):
    import torch

    cfg = tiny_cfg()
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer())
    checkpoint = trainer.save(tmp_path / "latest.json")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for missing in ("optimizer_state_dict", "rng_state"):
        modified = dict(payload)
        modified["meta"] = dict(payload["meta"])
        if missing == "rng_state":
            modified["meta"].pop(missing)
        else:
            modified.pop(missing)
        path = tmp_path / f"{missing}.pt"
        torch.save(modified, path)
        with pytest.raises(ValueError, match="optimizer|RNG"):
            Trainer.load(path, cfg, buffer=ReplayBuffer.load_npz(checkpoint.parent / "replay.npz"))


def test_resume_rejects_changed_optimizer_schedule(tmp_path):
    cfg = tiny_cfg()
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer())
    checkpoint = trainer.save(tmp_path / "latest.json")
    replay = ReplayBuffer.load_npz(checkpoint.parent / "replay.npz")
    with pytest.raises(ValueError, match="configuration differs"):
        Trainer.load(checkpoint, {**cfg, "learning_rate": cfg["learning_rate"] * 2}, buffer=replay)


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

def test_finetune_loads_weights_with_fresh_optimizer_and_empty_renju_buffer(tmp_path):
    from az.model import save_checkpoint

    cfg = tiny_cfg(rule_set="renju")
    source = AZNet(n_blocks=1, n_filters=8)
    checkpoint = tmp_path / "normal.pt"
    save_checkpoint(source, checkpoint, meta={"rule_set": "normal"})
    buffer = ReplayBuffer(capacity=32)

    trainer = Trainer.from_pretrained(checkpoint, cfg, buffer=buffer, seed=4)

    assert all((a.cpu() == b.cpu()).all() for a, b in zip(source.state_dict().values(), trainer.model.state_dict().values()))
    assert trainer.global_step == 0
    assert trainer.n_iter == 0
    assert len(buffer) == 0
    assert trainer.cfg["rule_set"] == "renju"


def test_inference_resolves_validated_pointer_and_accepts_explicit_checkpoint(tmp_path):
    from az.model import load_checkpoint, resolve_checkpoint_path

    pointer = tmp_path / "best.json"
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), tiny_cfg(), buffer=filled_buffer())
    checkpoint = trainer.save(pointer)
    assert resolve_checkpoint_path(pointer) == checkpoint
    assert resolve_checkpoint_path(checkpoint) == checkpoint
    model, _ = load_checkpoint(pointer, device="cpu")
    assert model.n_filters == 8
    (checkpoint.parent / "replay.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(pointer, device="cpu")


def test_bootstrap_restore_starts_at_schedule_iteration(tmp_path):
    cfg = tiny_cfg()
    source = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer())
    source.train_steps(cfg["train_steps_per_iter"])
    parent = source.save(tmp_path / "parent" / "latest.json")
    restored = Trainer.bootstrap_normal(parent, cfg, ReplayBuffer(32), "restore", device="cpu")
    assert restored.n_iter == 1
    assert len(range(restored.n_iter, cfg["n_iter"])) * cfg["train_steps_per_iter"] == (
        restored.total_steps - restored.global_step
    )

def test_bootstrap_restore_rejects_partial_schedule_iteration(tmp_path):
    cfg = tiny_cfg()
    source = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer())
    source.train_steps(1)
    parent = source.save(tmp_path / "parent" / "latest.json")
    with pytest.raises(ValueError, match="iteration boundary"):
        Trainer.bootstrap_normal(parent, cfg, ReplayBuffer(32), "restore", device="cpu")



def test_normal_bootstrap_reset_restore_and_persist_provenance(tmp_path):
    import torch
    from az.model import load_checkpoint

    cfg = tiny_cfg()
    source = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer())
    source.train_steps(cfg["train_steps_per_iter"])
    parent = source.save(tmp_path / "parent" / "latest.json")
    original = parent.read_bytes()
    for choice in ("reset", "restore"):
        buffer = ReplayBuffer(32)
        trainer = Trainer.bootstrap_normal(parent, cfg, buffer, choice, device="cpu")
        assert len(buffer) == 0
        assert trainer.n_iter == (0 if choice == "reset" else 1)
        assert trainer.global_step == (0 if choice == "reset" else cfg["train_steps_per_iter"])
        assert bool(trainer.optimizer.state) == (choice == "restore")
        assert all(torch.equal(a.cpu(), b.cpu()) for a, b in
                   zip(source.model.state_dict().values(), trainer.model.state_dict().values()))
        saved = trainer.save(tmp_path / choice / "latest.json")
        _, meta = load_checkpoint(saved, device="cpu")
        assert meta["bootstrap"] == trainer.bootstrap
        restored = Trainer.load(saved, cfg, ReplayBuffer.load_npz(saved.parent / "replay.npz"), device="cpu")
        assert restored.bootstrap == trainer.bootstrap
        assert restored.global_step == trainer.global_step
    assert parent.read_bytes() == original


def test_normal_bootstrap_rejects_unverified_optimizer_and_incompatible_inputs(tmp_path):
    from az.model import save_checkpoint

    cfg = tiny_cfg()
    parent = tmp_path / "parent.pt"
    save_checkpoint(AZNet(n_blocks=1, n_filters=8), parent,
                    meta={"rule_set": "normal", "config": cfg, "global_step": 1})
    with pytest.raises(ValueError, match="optimizer state"):
        Trainer.bootstrap_normal(parent, cfg, ReplayBuffer(32), "restore", device="cpu")
    with pytest.raises(ValueError, match="architecture"):
        Trainer.bootstrap_normal(parent, tiny_cfg(n_blocks=2), ReplayBuffer(32), "reset", device="cpu")
    with pytest.raises(ValueError, match="empty"):
        Trainer.bootstrap_normal(parent, cfg, filled_buffer(), "reset", device="cpu")
    with pytest.raises(ValueError, match="schedule"):
        Trainer.bootstrap_normal(parent, tiny_cfg(n_iter=3), ReplayBuffer(32), "restore", device="cpu")
