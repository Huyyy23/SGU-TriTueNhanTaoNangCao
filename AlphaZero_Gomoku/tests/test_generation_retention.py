import json
import shutil
from pathlib import Path

import pytest

from az.replay_buffer import ReplayBuffer
from az.trainer import Trainer, resolve_generation
from az.generation_retention import plan, apply
from config import DEFAULT_CONFIG


def tiny_cfg():
    return {**DEFAULT_CONFIG, "n_blocks": 1, "n_filters": 8, "batch_size": 16,
            "train_steps_per_iter": 1, "n_iter": 1, "eval_fixed_batch_size": 8,
            "rule_set": "normal", "buffer_size": 32}


def filled_buffer(n=16, seed=0):
    import numpy as np
    from gomoku.features import N_ACTIONS, N_PLANES, SIZE
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(capacity=64)
    for _ in range(n):
        planes = rng.integers(0, 3, size=(N_PLANES, SIZE, SIZE), dtype=np.uint8)
        pi = np.zeros(N_ACTIONS, dtype=np.float32)
        pi[rng.integers(N_ACTIONS)] = 1.0
        buf.add(planes, pi, float(rng.choice([-1.0, 1.0])), "normal")
    return buf


def make_generations(tmp_path, count=4):
    from az.model import AZNet
    pointer = tmp_path / "latest.json"
    cfg = tiny_cfg()
    trainer = Trainer(AZNet(n_blocks=1, n_filters=8), cfg, buffer=filled_buffer(), seed=1)
    first = trainer.save(pointer)
    first_id = first.parent.name
    for _ in range(count - 1):
        trainer.save(pointer)
    latest_id = json.loads(pointer.read_text())["generation"]
    return pointer, first_id, latest_id


def test_dry_run_does_not_delete(tmp_path):
    pointer, first_id, latest_id = make_generations(tmp_path, 4)
    gens_before = sorted((tmp_path / "generations").iterdir())
    result = plan(tmp_path, dry_run=True)
    assert result["dry_run"] is True
    assert len(result["candidates"]) >= 1
    assert (tmp_path / "generations").exists()
    assert sorted((tmp_path / "generations").iterdir()) == gens_before


def test_apply_keeps_active_best_previous_and_deletes_old(tmp_path):
    pointer, first_id, latest_id = make_generations(tmp_path, 4)
    # Mark first generation as best.
    (tmp_path / "best.json").write_text(json.dumps({"generation": first_id}))
    # Resolve latest manifest previous.
    latest_dir = tmp_path / "generations" / latest_id
    manifest = json.loads((latest_dir / "manifest.json").read_text())
    previous_id = manifest.get("commit_previous") or manifest.get("previous")
    assert previous_id is not None

    result = apply(tmp_path, dry_run=False)
    assert result["freed_bytes"] >= 0
    remaining = {p.name for p in (tmp_path / "generations").iterdir()}
    assert latest_id in remaining
    assert first_id in remaining
    if previous_id in {p.name for p in (tmp_path / "generations").iterdir() if True}:
        pass
    assert previous_id in remaining
    # resume still works
    checkpoint, replay = resolve_generation(pointer)
    assert checkpoint.is_file() and replay.is_file()


def test_cleanup_failure_stops_and_keeps_candidates(tmp_path, monkeypatch):
    pointer, first_id, latest_id = make_generations(tmp_path, 4)
    (tmp_path / "best.json").write_text(json.dumps({"generation": first_id}))

    original = shutil.rmtree

    def fail(path):
        raise OSError("simulated failure")

    monkeypatch.setattr(shutil, "rmtree", fail)
    result = apply(tmp_path, dry_run=False)
    monkeypatch.setattr(shutil, "rmtree", original)
    assert result["deleted"] == []
    assert result["errors"]
    assert len(list((tmp_path / "generations").iterdir())) == 4


def test_protected_reference_keeps_generation(tmp_path):
    pointer, first_id, latest_id = make_generations(tmp_path, 4)
    # Create an extra protected file referencing a non-kept generation.
    extra = tmp_path / "protected.txt"
    candidates_before = plan(tmp_path)["candidates"]
    if not candidates_before:
        pytest.skip("no candidates")
    protected_gen = candidates_before[0]
    extra.write_text(f"reference {protected_gen}")
    result = plan(tmp_path, protected_paths=[extra])
    assert protected_gen not in result["candidates"]
