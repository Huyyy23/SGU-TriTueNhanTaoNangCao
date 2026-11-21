"""Điều tra mâu thuẫn: speedup end-to-end 2.38x nhưng profiling cho thấy inference chiếm 92-95%.

Hypothesis: Profiling cũ đo batch=1 (mỗi nước một lần infer), không đại diện cho
end-to-end batched (parallel_games=16, mean_batch ~7.5). Có thể có "missing time"
ở Python overhead per tick/receive/request mà profiling cũ không bắt được.

Script instrument model.predict de tich luu thoi gian inference thuc tat (CUDA sync),
chay run_games voi wrapper, sau do tinh:
- wall time (end-to-end)
- tong thoi gian inference (tich luu tu wrapper)
- non-inference time = wall - inference (MCTS request/receive + Python overhead + khac)
So sanh Python vs C++ cung checkpoint, cung n_simulations, cung parallel_games.

Khong sua ma san pham, chi tao script throwaway moi.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from az.model import load_checkpoint, AZNet
from az.selfplay import run_games
from config import DEFAULT_CONFIG
from gomoku.rules import get_rules

CKPT = "checkpoints/latest.pt"
RULE_SET = "normal"
N_SIMULATIONS = 200
PARALLEL_GAMES = 16
N_GAMES = 8
SEEDS = [1, 2, 3]
REPEATS = 2


class TimingModelWrapper:
    """Wrap AZNet de tich luu thoi gian inference thuc tat (co CUDA sync)."""
    def __init__(self, model):
        self.model = model
        self.infer_sec = 0.0
        self.calls = 0

    def predict(self, features):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        probs, values = self.model.predict(features)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.infer_sec += time.perf_counter() - t0
        self.calls += 1
        return probs, values


def make_cfg(backend: str, seed: int) -> dict:
    return {
        **DEFAULT_CONFIG,
        "rule_set": RULE_SET,
        "n_simulations": N_SIMULATIONS,
        "parallel_games": PARALLEL_GAMES,
        "temp_moves": 0,
        "mcts_backend": backend,
        "patterns_backend": "python",
        "log_moves": False,
        "backend_metadata": {"mcts": backend, "patterns": "python"},
    }


def warmup_gpu(model):
    if not torch.cuda.is_available():
        return
    dummy = np.zeros((1, 4, 11, 11), dtype=np.float32)
    for _ in range(5):
        model.predict(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_once(model, rules, cfg, seed: int):
    wrapper = TimingModelWrapper(model)
    samples = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    records = list(run_games(wrapper, rules, cfg, N_GAMES, seed=seed, samples_out=samples))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    infer = wrapper.infer_sec
    non_infer = wall - infer
    total_moves = sum(int(r.moves_count) for r in records)
    return {
        "games": len(records),
        "moves": total_moves,
        "wall_sec": wall,
        "infer_sec": infer,
        "non_infer_sec": non_infer,
        "infer_calls": wrapper.calls,
        "mean_batch": float(np.mean([r.mean_batch for r in records])) if records else 0.0,
        "infer_per_call_ms": (infer / wrapper.calls * 1e3) if wrapper.calls else 0.0,
        "non_infer_per_move_ms": (non_infer / total_moves * 1e3) if total_moves else 0.0,
        "infer_pct_of_wall": (infer / wall * 100) if wall > 0 else 0.0,
        "non_infer_pct_of_wall": (non_infer / wall * 100) if wall > 0 else 0.0,
    }


def summarize(vals):
    return {
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rules = get_rules(RULE_SET)
    model, meta = load_checkpoint(CKPT, device=device)
    model.eval()
    torch.set_num_threads(1)
    warmup_gpu(model)
    print(f"Checkpoint: {CKPT} | Device: {device} | cuda: {torch.cuda.is_available()}")
    print(f"Config: n_simulations={N_SIMULATIONS}, parallel_games={PARALLEL_GAMES}, "
          f"games={N_GAMES}, seeds={SEEDS}, repeats={REPEATS}")
    print("=" * 100)

    rows = []
    for backend in ("python", "cpp"):
        for rep in range(REPEATS):
            for seed in SEEDS:
                cfg = make_cfg(backend, seed)
                r = run_once(model, rules, cfg, seed)
                rows.append({"backend": backend, "rep": rep, "seed": seed, **r})

    print("\n=== End-to-end phase timing (Python vs C++), cung checkpoint/trong so ===")
    print(f"{'backend':<9} {'rep':<4} {'seed':<5} {'wall(s)':<9} {'infer(s)':<9} {'noninf(s)':<10} "
          f"{'infer%':<7} {'noninf%':<8} {'inf_calls':<9} {'inf/call_ms':<11} {'noninf/move_ms':<13}")
    for r in rows:
        print(f"{r['backend']:<9} {r['rep']:<4} {r['seed']:<5} {r['wall_sec']:<9.3f} {r['infer_sec']:<9.3f} "
              f"{r['non_infer_sec']:<10.3f} {r['infer_pct_of_wall']:<7.1f} {r['non_infer_pct_of_wall']:<8.1f} "
              f"{r['infer_calls']:<9} {r['infer_per_call_ms']:<11.3f} {r['non_infer_per_move_ms']:<13.3f}")

    print("\n=== Aggregate (median/min/max/std qua rep x seed) ===")
    for backend in ("python", "cpp"):
        sub = [r for r in rows if r["backend"] == backend]
        wall = summarize([r["wall_sec"] for r in sub])
        infer = summarize([r["infer_sec"] for r in sub])
        noninf = summarize([r["non_infer_sec"] for r in sub])
        gps = summarize([r["wall_sec"] and (r["games"] / r["wall_sec"]) for r in sub])
        print(f"\n{backend.upper()}:")
        print(f"  wall(s):        med={wall['median']:.3f} min={wall['min']:.3f} max={wall['max']:.3f} std={wall['std']:.3f}")
        print(f"  infer(s):       med={infer['median']:.3f} min={infer['min']:.3f} max={infer['max']:.3f} std={infer['std']:.3f}")
        print(f"  non_infer(s):   med={noninf['median']:.3f} min={noninf['min']:.3f} max={noninf['max']:.3f} std={noninf['std']:.3f}")
        print(f"  games/s:        med={gps['median']:.3f} min={gps['min']:.3f} max={gps['max']:.3f} std={gps['std']:.3f}")
        print(f"  infer% of wall: med={statistics.median([r['infer_pct_of_wall'] for r in sub]):.1f}%")
        print(f"  non_infer%:     med={statistics.median([r['non_infer_pct_of_wall'] for r in sub]):.1f}%")

    py_wall = [r["wall_sec"] for r in rows if r["backend"] == "python"]
    cpp_wall = [r["wall_sec"] for r in rows if r["backend"] == "cpp"]
    py_noninf = [r["non_infer_sec"] for r in rows if r["backend"] == "python"]
    cpp_noninf = [r["non_infer_sec"] for r in rows if r["backend"] == "cpp"]
    py_infer = [r["infer_sec"] for r in rows if r["backend"] == "python"]
    cpp_infer = [r["infer_sec"] for r in rows if r["backend"] == "cpp"]

    print("\n=== So sanh thuc te (median) ===")
    print(f"Python wall med : {statistics.median(py_wall):.3f}s")
    print(f"C++   wall med  : {statistics.median(cpp_wall):.3f}s")
    print(f"Speedup wall    : {statistics.median(py_wall) / statistics.median(cpp_wall):.3f}x")
    print(f"Python infer med: {statistics.median(py_infer):.3f}s")
    print(f"C++   infer med : {statistics.median(cpp_infer):.3f}s")
    print(f"Infer ratio     : {statistics.median(py_infer) / statistics.median(cpp_infer):.3f}x")
    print(f"Python noninf med: {statistics.median(py_noninf):.3f}s")
    print(f"C++   noninf med : {statistics.median(cpp_noninf):.3f}s")
    print(f"Non-infer ratio : {statistics.median(py_noninf) / statistics.median(cpp_noninf):.3f}x")
    print(f"Non-infer saved : {statistics.median(py_noninf) - statistics.median(cpp_noninf):.3f}s "
          f"({(1 - statistics.median(cpp_noninf) / statistics.median(py_noninf)) * 100:.1f}% reduction)")


if __name__ == "__main__":
    main()
