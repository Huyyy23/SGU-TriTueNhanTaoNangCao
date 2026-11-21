"""Benchmark thực tế hiệu năng Python/C++ trong Gomoku AlphaZero self-play.

Không sửa mã sản phẩm, không chạy training, không đổi checkpoint/trọng số/SQLite/dữ liệu.
Chạy bằng: .venv/Scripts/python.exe scripts/bench_real_world.py

Đo:
- games/second, elapsed time, inference calls, batch efficiency (mean batch)
- overhead pybind11 (wrapper scalar cho MCTS.search)
- Profiling riêng: MCTS request/receive, feature extraction (board_to_planes), inference (CUDA sync), Python overhead
- CPU utilization; GPU utilization qua nvidia-smi (không cài pynvml mới)
Cùng checkpoint, cùng n_simulations (200), cùng parallel_games (16), nhiều seed, nhiều lần lặp.
Báo cáo median, min, max, std.
"""
from __future__ import annotations

import statistics
import subprocess
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
from az.mcts import MCTS, NativeMCTS
from config import DEFAULT_CONFIG
from gomoku.board import Board, SIZE
from gomoku.features import board_to_planes
from gomoku.rules import get_rules

CKPT = "checkpoints/latest.pt"
RULE_SET = "normal"
N_SIMULATIONS = 200
PARALLEL_GAMES = 16
N_GAMES = 8
SEEDS = [1, 2, 3]
REPEATS = 2


class ScalarEvaluator:
    """Wrap model.predict để trả về scalar (policy, value) dùng cho MCTS.search."""
    def __init__(self, model):
        self.model = model

    def predict(self, planes_b4_11_11):
        probs, values = self.model.predict(planes_b4_11_11)
        return probs[0], float(values[0])


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


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_snapshot() -> tuple[float, float]:
    """(util_gpu %, mem_used MB) qua nvidia-smi subprocess, hoặc nan nếu không có."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            parts = out.stdout.strip().splitlines()[0].split(",")
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return float("nan"), float("nan")


def cpu_percent() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=0.0)
    except Exception:
        return float("nan")


def run_end_to_end(model, rules, cfg, seed: int, device):
    """Chạy run_games, trả về dict chỉ số. Đồng bộ CUDA sau khi xong."""
    samples = []
    cuda_sync()
    t0 = time.perf_counter()
    records = list(run_games(model, rules, cfg, N_GAMES, seed=seed, samples_out=samples))
    cuda_sync()
    wall = time.perf_counter() - t0
    nn_calls = sum(int(r.nn_calls) for r in records)
    mean_batches = [float(r.mean_batch) for r in records if r.mean_batch]
    return {
        "games": len(records),
        "moves": sum(int(r.moves_count) for r in records),
        "wall_sec": wall,
        "games_per_sec": len(records) / wall if wall > 0 else 0.0,
        "moves_per_sec": sum(int(r.moves_count) for r in records) / wall if wall > 0 else 0.0,
        "inference_calls": nn_calls,
        "mean_batch": (sum(mean_batches) / len(mean_batches) if mean_batches else 0.0),
    }


def profile_components(model, rules, backend: str, device, n_moves: int = 40, seed: int = 7) -> dict:
    """Profile tách biệt trên một ván ngắn: MCTS, feature, inference (sync), receive."""
    rng = np.random.default_rng(seed)
    board = Board()
    mcts_cls = NativeMCTS if backend == "cpp" else MCTS
    mcts = mcts_cls(
        rules=rules, n_simulations=N_SIMULATIONS, c_puct=1.5,
        dirichlet_alpha=0.15, dirichlet_eps=0.25, root_noise=False, rng=rng,
    )
    mcts.reset(board)

    t_mcts = t_feature = t_infer = t_receive = 0.0
    n_infer = 0
    moves_done = 0

    while moves_done < n_moves and not rules.is_terminal(board):
        t0 = time.perf_counter()
        req = mcts.request()
        t_mcts += time.perf_counter() - t0
        if req is None:
            break
        t0 = time.perf_counter()
        planes = board_to_planes(board, rules, board.current_player)
        t_feature += time.perf_counter() - t0
        batch = planes[np.newaxis, ...].astype(np.float32)
        cuda_sync()
        t0 = time.perf_counter()
        probs, values = model.predict(batch)
        cuda_sync()
        t_infer += time.perf_counter() - t0
        n_infer += 1
        t0 = time.perf_counter()
        mcts.receive(probs[0], float(values[0]))
        t_receive += time.perf_counter() - t0
        action = int(mcts.select_move(0.0)) if mcts.visit_counts().sum() > 0 else int(rng.choice(rules.legal_moves(board)))
        board.play(*divmod(action, SIZE))
        if not rules.is_terminal(board):
            mcts.reset(board)
        moves_done += 1

    total = t_mcts + t_feature + t_infer + t_receive
    return {
        "backend": backend,
        "mcts_ms": t_mcts * 1e3,
        "feature_ms": t_feature * 1e3,
        "infer_ms": t_infer * 1e3,
        "receive_ms": t_receive * 1e3,
        "infer_per_call_ms": (t_infer / n_infer * 1e3) if n_infer else 0.0,
        "n_infer": n_infer,
        "moves": moves_done,
        "total_ms": total * 1e3,
        "pct_mcts": (t_mcts / total * 100) if total > 0 else 0.0,
        "pct_feature": (t_feature / total * 100) if total > 0 else 0.0,
        "pct_infer": (t_infer / total * 100) if total > 0 else 0.0,
        "pct_receive": (t_receive / total * 100) if total > 0 else 0.0,
    }


def summarize(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rules = get_rules(RULE_SET)
    model, meta = load_checkpoint(CKPT, device=device)
    model.eval()
    torch.set_num_threads(1)
    print(f"Checkpoint: {CKPT}")
    print(f"Device: {device} | cuda: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: n_simulations={N_SIMULATIONS}, parallel_games={PARALLEL_GAMES}, "
          f"games={N_GAMES}, seeds={SEEDS}, repeats={REPEATS}")
    print("=" * 90)

    rows = []
    for backend in ("python", "cpp"):
        for rep in range(REPEATS):
            for seed in SEEDS:
                cfg = make_cfg(backend, seed)
                gpu_before = gpu_snapshot()
                cpu_before = cpu_percent()
                r = run_end_to_end(model, rules, cfg, seed, device)
                gpu_after = gpu_snapshot()
                cpu_after = cpu_percent()
                rows.append({
                    "backend": backend, "rep": rep, "seed": seed, **r,
                    "cpu_before": cpu_before, "cpu_after": cpu_after,
                    "gpu_util_before": gpu_before[0], "gpu_util_after": gpu_after[0],
                    "gpu_mem_before": gpu_before[1], "gpu_mem_after": gpu_after[1],
                })

    print("\n=== End-to-end self-play (Python vs C++), cùng checkpoint/trọng số ===")
    print(f"{'backend':<9} {'rep':<4} {'seed':<5} {'games':<6} {'moves':<7} {'wall(s)':<9} "
          f"{'g/s':<8} {'m/s':<8} {'inf_calls':<10} {'mean_batch':<10} {'cpu%':<7} {'gpu%':<8}")
    for r in rows:
        print(f"{r['backend']:<9} {r['rep']:<4} {r['seed']:<5} {r['games']:<6} {r['moves']:<7} "
              f"{r['wall_sec']:<9.3f} {r['games_per_sec']:<8.3f} {r['moves_per_sec']:<8.1f} "
              f"{r['inference_calls']:<10} {r['mean_batch']:<10.2f} {r['cpu_after']:<7.1f} {r['gpu_util_after']:<8.1f}")

    print("\n=== Aggregate (median / min / max / std qua rep x seed) ===")
    for backend in ("python", "cpp"):
        sub = [r for r in rows if r["backend"] == backend]
        walls = [r["wall_sec"] for r in sub]
        gps = [r["games_per_sec"] for r in sub]
        mps = [r["moves_per_sec"] for r in sub]
        inf = [r["inference_calls"] for r in sub]
        batch = [r["mean_batch"] for r in sub]
        sw = summarize(walls); sg = summarize(gps); sm = summarize(mps)
        print(f"{backend:<9} wall(s): med={sw['median']:.3f} min={sw['min']:.3f} max={sw['max']:.3f} std={sw['std']:.3f}")
        print(f"{'':<9}  games/s: med={sg['median']:.3f} min={sg['min']:.3f} max={sg['max']:.3f} std={sg['std']:.3f}")
        print(f"{'':<9}  moves/s: med={sm['median']:.1f} min={sm['min']:.1f} max={sm['max']:.1f} std={sm['std']:.1f}")
        print(f"{'':<9}  inf_calls: med={summarize(inf)['median']:.0f} mean_batch: med={summarize(batch)['median']:.2f}")
    py_gps = [r["games_per_sec"] for r in rows if r["backend"] == "python"]
    cpp_gps = [r["games_per_sec"] for r in rows if r["backend"] == "cpp"]
    if py_gps and cpp_gps:
        py_med = statistics.median(py_gps)
        cpp_med = statistics.median(cpp_gps)
        print(f"\nSpeedup C++/Python (games/s, median): {cpp_med / py_med:.3f}x")

    # Component profiling
    print("\n=== Component profiling (một ván ngắn, 40 nước, n_simulations=200, CUDA sync) ===")
    print(f"{'backend':<9} {'mcts_ms':<9} {'feature_ms':<11} {'infer_ms':<9} {'receive_ms':<11} "
          f"{'infer/call_ms':<12} {'n_infer':<8} {'moves':<6} {'pct_infer':<10}")
    for backend in ("python", "cpp"):
        p = profile_components(model, rules, backend, device, seed=11)
        print(f"{backend:<9} {p['mcts_ms']:<9.2f} {p['feature_ms']:<11.3f} {p['infer_ms']:<9.3f} "
              f"{p['receive_ms']:<11.3f} {p['infer_per_call_ms']:<12.3f} {p['n_infer']:<8} {p['moves']:<6} "
              f"{p['pct_infer']:<10.1f}%")
        print(f"{'':<9}  mcts={p['pct_mcts']:.1f}% feature={p['pct_feature']:.1f}% infer={p['pct_infer']:.1f}% receive={p['pct_receive']:.1f}%")

    print("\n=== Overhead pybind11 (MCTS tree only, 200 sims, same board, model.predict ndarray API) ===")
    b1, b2 = Board(), Board()
    py = MCTS(rules=rules, n_simulations=N_SIMULATIONS, c_puct=1.5,
              dirichlet_alpha=0.15, dirichlet_eps=0.25, root_noise=False, rng=np.random.default_rng(3))
    py.reset(b1)
    cuda_sync(); t0 = time.perf_counter()
    py.search(b1, model.predict)
    cuda_sync(); t_py = time.perf_counter() - t0
    cpp = NativeMCTS(rules=rules, n_simulations=N_SIMULATIONS, c_puct=1.5,
                     dirichlet_alpha=0.15, dirichlet_eps=0.25, root_noise=False, rng=np.random.default_rng(3))
    cpp.reset(b2)
    cuda_sync(); t0 = time.perf_counter()
    cpp.search(b2, model.predict)
    cuda_sync(); t_cpp = time.perf_counter() - t0
    print(f"Python MCTS tree: {t_py*1e3:.2f} ms | C++ MCTS tree: {t_cpp*1e3:.2f} ms | "
          f"speedup: {t_py/max(t_cpp,1e-9):.2f}x | visits equal: {np.array_equal(py.visit_counts(), cpp.visit_counts())}")


if __name__ == "__main__":
    main()
