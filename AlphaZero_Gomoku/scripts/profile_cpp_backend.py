"""Profiling workload đại diện cho Python/C++ dual backend.

Đo thời gian: MCTS (python & cpp), board/rules, feature extraction, PyTorch inference,
và overhead Python<->C++ (request/receive) trên một ván self-play ngắn.
Chạy bằng `.venv/Scripts/python.exe scripts/profile_cpp_backend.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from az.model import AZNet
from az.selfplay import run_games, run_games_parallel
from az.mcts import MCTS, NativeMCTS
from config import DEFAULT_CONFIG
from gomoku.board import Board, SIZE
from gomoku.features import board_to_planes
from gomoku.rules import get_rules


class TimingEvaluator:
    """Fake evaluator that records inference latency like the real model."""
    def __init__(self, n_blocks=2, n_filters=16):
        self.model = AZNet(n_blocks=n_blocks, n_filters=n_filters)
        self.infer_sec = 0.0
        self.calls = 0

    def predict(self, features):
        if hasattr(features, "planes"):
            features = features.planes
        features = np.asarray(features)
        single = features.ndim == 3
        if single:
            features = features[np.newaxis, ...]
        t0 = time.perf_counter()
        probs, values = self.model.predict(features)
        self.infer_sec += time.perf_counter() - t0
        self.calls += 1
        if single:
            return probs[0], float(values[0])
        return probs, values


def elapsed(fn, *a, **k):
    t0 = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - t0


def main() -> None:
    rules = get_rules("normal")
    board = Board()
    model = TimingEvaluator()
    n_sim = 50

    # --- MCTS python vs cpp on one move (fixed board, same evaluator) ---
    py = MCTS(rules=rules, n_simulations=n_sim, c_puct=1.5,
              dirichlet_alpha=0.15, dirichlet_eps=0.25, root_noise=False,
              rng=np.random.default_rng(1))
    b1 = Board()
    _, t_py = elapsed(py.search, b1, model.predict)

    cpp = NativeMCTS(rules=rules, n_simulations=n_sim, c_puct=1.5,
                     dirichlet_alpha=0.15, dirichlet_eps=0.25, root_noise=False,
                     rng=np.random.default_rng(1))
    b2 = Board()
    _, t_cpp = elapsed(cpp.search, b2, model.predict)

    print("=== MCTS one move, simulations={} ===".format(n_sim))
    print(f"  python MCTS : {t_py*1e3:.2f} ms, simulations_done={py.simulations_done}")
    print(f"  cpp   MCTS  : {t_cpp*1e3:.2f} ms, simulations_done={cpp.simulations_done}")
    if t_py > 0:
        print(f"  speedup     : {t_py / max(t_cpp, 1e-9):.2f}x")
    print(f"  visits equal: {np.array_equal(py.visit_counts(), cpp.visit_counts())}")
    print(f"  root value  : py={py.root_value():.4f} cpp={cpp.root_value():.4f}")

    # --- board/rules / features / inference micro-benchmarks ---
    n = 2000

    def play_undo_all():
        for i in range(n):
            b = Board()
            r, c = divmod(i % 121, SIZE)
            b.play(r, c)
            b.undo()

    _, t_play = elapsed(play_undo_all)
    rules = get_rules("normal")
    b = Board()
    _, t_legal = elapsed(lambda: [rules.legal_moves(b) for _ in range(n)])
    _, t_win = elapsed(lambda: [rules.is_win(b) for _ in range(n)])
    _, t_term = elapsed(lambda: [rules.is_terminal(b) for _ in range(n)])
    planes = board_to_planes(b, rules, b.current_player)
    _, t_planes = elapsed(lambda: board_to_planes(b, rules, b.current_player))
    _, t_infer = elapsed(lambda: model.model.predict(np.stack([planes] * 32)))

    print("\n=== Micro benchmarks (n={}) ===".format(n))
    print(f"  play/undo x{n}        : {t_play*1e3:.2f} ms")
    print(f"  legal_moves x{n}      : {t_legal*1e3:.2f} ms")
    print(f"  is_win x{n}           : {t_win*1e3:.2f} ms")
    print(f"  is_terminal x{n}      : {t_term*1e3:.2f} ms")
    print(f"  board_to_planes x1    : {t_planes*1e3:.2f} ms")
    print(f"  model.predict batch32 : {t_infer*1e3:.2f} ms ({t_infer/32*1e3:.3f} ms per call)")

    # --- Self-play end-to-end: python vs cpp ---
    cfg = {**DEFAULT_CONFIG, "n_simulations": 40, "temp_moves": 0,
           "parallel_games": 1, "rule_set": "normal", "log_moves": False}
    samples_py, samples_cpp = [], []
    _, t_py_game = elapsed(lambda: list(run_games(model, rules, cfg, 2, seed=7, samples_out=samples_py)))
    _, t_cpp_game = elapsed(lambda: list(run_games(model, rules, {**cfg, "mcts_backend": "cpp"}, 2,
                                                  seed=7, samples_out=samples_cpp)))

    print("\n=== Self-play 2 games, simulations=40 ===")
    print(f"  python backend : {t_py_game*1e3:.2f} ms, nn_calls={model.calls}")
    print(f"  cpp   backend  : {t_cpp_game*1e3:.2f} ms")
    print(f"  samples py/cpp : {len(samples_py)}/{len(samples_cpp)}")
    print(f"  z equal        : all(z_p == z_c)",
          all(z_p == z_c for (_, _, z_p), (_, _, z_c) in zip(samples_py, samples_cpp)))
    print(f"  pi close       : all(np.allclose(p_p, p_c, atol=1e-6))",
          all(np.allclose(p_p, p_c, atol=1e-6) for (_, p_p, _), (_, p_c, _) in zip(samples_py, samples_cpp)))

    # --- Parallel workers cpp ---
    model2 = TimingEvaluator()
    cfg2 = {**cfg, "n_blocks": 2, "n_filters": 16, "parallel_games": 1}
    with ProcessPoolExecutor(max_workers=2) as pool:
        _, t_par = elapsed(lambda: run_games_parallel(pool, model2.model, rules,
                                                      {**cfg2, "mcts_backend": "cpp"}, 4,
                                                      seed=3, workers=2))
    print(f"\n=== Parallel cpp self-play 4 games / 2 workers ===")
    print(f"  wall time : {t_par*1e3:.2f} ms")


if __name__ == "__main__":
    main()
