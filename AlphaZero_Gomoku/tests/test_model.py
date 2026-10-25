import sys
import time
from pathlib import Path
import numpy as np
import pytest
import torch

# Ensure AlphaZero_Gomoku root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.model import AZNet, load_checkpoint, save_checkpoint


def test_output_shapes():
    model = AZNet(n_blocks=2, n_filters=32)
    model.eval()

    # Forward check
    x = torch.randn(3, 4, 11, 11, dtype=torch.float32)
    logits, values = model(x)
    assert logits.shape == (3, 121)
    assert values.shape == (3, 1)
    assert torch.all(values >= -1.0) and torch.all(values <= 1.0)

    # Predict check
    features = np.zeros((3, 4, 11, 11), dtype=np.uint8)
    features[:, 0, 5, 5] = 1
    probs, vals = model.predict(features)
    assert probs.shape == (3, 121)
    assert vals.shape == (3,)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(vals >= -1.0) and np.all(vals <= 1.0)


def test_invalid_input_shapes():
    model = AZNet(n_blocks=1, n_filters=16)
    with pytest.raises(ValueError):
        # 3D instead of 4D
        model.predict(np.zeros((4, 11, 11), dtype=np.uint8))

    with pytest.raises(ValueError):
        # Wrong board size
        model.predict(np.zeros((1, 4, 9, 9), dtype=np.uint8))

    with pytest.raises(TypeError):
        # Non-numpy
        model.predict([[0]])


def test_batch_consistency():
    model = AZNet(n_blocks=2, n_filters=32)
    model.eval()

    rng = np.random.default_rng(42)
    features_batch = rng.integers(0, 2, size=(4, 4, 11, 11), dtype=np.uint8)

    probs_batch, vals_batch = model.predict(features_batch)

    for i in range(4):
        single_feature = features_batch[i : i + 1]
        probs_single, vals_single = model.predict(single_feature)
        np.testing.assert_allclose(probs_batch[i], probs_single[0], atol=1e-5)
        np.testing.assert_allclose(vals_batch[i], vals_single[0], atol=1e-5)


def test_predict_deterministic():
    model = AZNet(n_blocks=2, n_filters=32)
    model.eval()

    features = np.ones((2, 4, 11, 11), dtype=np.uint8)
    p1, v1 = model.predict(features)
    p2, v2 = model.predict(features)

    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(v1, v2)


def test_checkpoint_roundtrip(tmp_path):
    model = AZNet(n_blocks=3, n_filters=32)
    meta = {
        "n_iter": 5,
        "rule_set": "normal",
        "config": {"lr": 1e-3},
    }
    ckpt_path = tmp_path / "test_model.pt"
    save_checkpoint(model, ckpt_path, meta=meta)

    loaded_model, loaded_meta = load_checkpoint(ckpt_path, device="cpu")
    assert set(loaded_meta) >= {"format_version", "n_blocks", "n_filters", "n_iter", "rule_set", "config"}
    assert loaded_meta["format_version"] == 1
    assert loaded_meta["n_blocks"] == 3
    assert loaded_meta["n_filters"] == 32
    assert loaded_meta["n_iter"] == 5
    assert loaded_meta["rule_set"] == "normal"
    assert loaded_meta["config"] == {"lr": 1e-3}

    # Compare predictions
    features = np.zeros((1, 4, 11, 11), dtype=np.uint8)
    p_orig, v_orig = model.predict(features)
    p_load, v_load = loaded_model.predict(features)

    np.testing.assert_allclose(p_orig, p_load, atol=1e-5)
    np.testing.assert_allclose(v_orig, v_load, atol=1e-5)


def test_predict_restores_training_mode():
    model = AZNet(n_blocks=1, n_filters=16)
    model.train()
    model.predict(np.zeros((1, 4, 11, 11), dtype=np.uint8))
    assert model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_checkpoint_saved_on_gpu_loads_on_cpu(tmp_path):
    model = AZNet(n_blocks=2, n_filters=16).to("cuda")
    ckpt_path = tmp_path / "gpu_model.pt"
    save_checkpoint(model, ckpt_path, meta={"n_iter": 1, "rule_set": "normal"})

    loaded_model, loaded_meta = load_checkpoint(ckpt_path, device="cpu")
    assert next(loaded_model.parameters()).device.type == "cpu"
    assert loaded_meta["n_iter"] == 1
    assert loaded_meta["rule_set"] == "normal"

    features = np.zeros((2, 4, 11, 11), dtype=np.uint8)
    features[0, 0, 3, 3] = 1
    features[1, 1, 7, 7] = 1
    p_gpu, v_gpu = model.predict(features)
    p_cpu, v_cpu = loaded_model.predict(features)
    np.testing.assert_allclose(p_gpu, p_cpu, atol=1e-5)
    np.testing.assert_allclose(v_gpu, v_cpu, atol=1e-5)


def test_cpu_fallback():
    model = AZNet(n_blocks=2, n_filters=16)
    model.to("cpu")
    features = np.zeros((2, 4, 11, 11), dtype=np.uint8)
    probs, vals = model.predict(features)
    assert probs.shape == (2, 121)
    assert vals.shape == (2,)


def test_benchmark_batch_16_latency():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AZNet(n_blocks=6, n_filters=64).to(device)
    model.eval()

    batch_size = 16
    features = np.zeros((batch_size, 4, 11, 11), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        _ = model.predict(features)

    if device.type == "cuda":
        torch.cuda.synchronize()

    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        _ = model.predict(features)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    ms_per_batch = (elapsed / iterations) * 1000.0
    print(f"\n[Benchmark] Device: {device.type}, Batch: {batch_size}, Latency: {ms_per_batch:.2f} ms/batch")
    assert ms_per_batch > 0
