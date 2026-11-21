#!/usr/bin/env python3
"""Compare pattern backends in isolated processes; print JSON to stdout.

Usage: python scripts/bench_native_patterns.py --checkpoint checkpoints/best.json
Each matchup plays three games per colour. No benchmark artifacts are created.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261115


def peak_ram_bytes():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                    "PagefileUsage", "PeakPagefileUsage",
                )
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize
        return None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def worker(backend, checkpoint):
    sys.path.insert(0, str(ROOT))
    start = time.perf_counter()
    import numpy as np
    import torch
    from gomoku.board import BLACK, WHITE, Board, SIZE
    import gomoku.patterns as patterns
    from gomoku.rules.normal import NormalRules
    from players.factory import create_player

    import_seconds = time.perf_counter() - start
    if patterns._backend != backend or (patterns._native is not None) != (backend == "native"):
        raise RuntimeError(f"requested {backend} backend was not selected")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rules = NormalRules()
    board = Board()
    for move in ((5, 5), (5, 6), (6, 5), (4, 5), (6, 6), (4, 6)):
        board.play(*move)
    cells = [(r, c) for r in range(SIZE) for c in range(SIZE) if board.at(r, c) == 0]
    start = time.perf_counter()
    for r, c in cells:
        patterns.find_patterns(board, r, c, BLACK)
    warmup_seconds = time.perf_counter() - start
    samples = []
    for _ in range(10):
        for r, c in cells:
            start = time.perf_counter_ns()
            patterns.find_patterns(board, r, c, BLACK)
            samples.append(time.perf_counter_ns() - start)
    samples.sort()
    latency = {"calls": len(samples), "mean_us": sum(samples) / len(samples) / 1000,
               "median_us": samples[len(samples) // 2] / 1000,
               "p95_us": samples[int((len(samples) - 1) * .95)] / 1000}
    start = time.perf_counter()
    az = create_player("az", rules, checkpoint=checkpoint, seed=SEED, n_simulations=200)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    az.select_move(Board())
    az_warmup_seconds = time.perf_counter() - start
    results = {}
    for opponent in ("heuristic", "az"):
        games = []
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        for repetition in range(3):
            for alpha_colour in (BLACK, WHITE):
                random.seed(SEED + repetition)
                np.random.seed(SEED + repetition)
                torch.manual_seed(SEED + repetition)
                alpha = create_player("alphabeta", rules, depth=1)
                other = (create_player("az", rules, checkpoint=checkpoint, seed=SEED + repetition, n_simulations=200)
                         if opponent == "az" else create_player("heuristic", rules))
                players = {alpha_colour: alpha, WHITE if alpha_colour == BLACK else BLACK: other}
                board = Board()
                while True:
                    colour = board.current_player
                    r, c = players[colour].select_move(board)
                    if r * SIZE + c not in rules.legal_moves(board):
                        raise RuntimeError(f"illegal move {(r, c)}")
                    board.play(r, c)
                    if rules.is_win(board):
                        winner = colour
                        break
                    if not rules.legal_moves(board):
                        winner = None
                        break
                games.append({"repetition": repetition, "alphabeta_colour": alpha_colour,
                              "winner_colour": winner, "moves": len(board.history)})
        wall, cpu = time.perf_counter() - wall_start, time.process_time() - cpu_start
        results[opponent] = {"games": games, "wall_seconds": wall, "cpu_seconds": cpu,
                             "cpu_percent_one_core": 100 * cpu / wall,
                             "games_per_minute": 60 * len(games) / wall,
                             "moves_per_second": sum(game["moves"] for game in games) / wall,
                             "peak_process_ram_bytes": peak_ram_bytes()}
    return {"backend": backend, "native_module": getattr(patterns._native, "__file__", None),
            "import_seconds": import_seconds, "build_seconds": None,
            "build_note": "prebuilt extension; build not performed",
            "pattern_warmup_seconds": warmup_seconds, "az_load_seconds": load_seconds,
            "az_warmup_seconds": az_warmup_seconds, "pattern_latency": latency,
            "matchups": results, "peak_process_ram_bytes": peak_ram_bytes()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "best.json")
    parser.add_argument("--worker", choices=("python", "native"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    from az.model import resolve_checkpoint_path
    checkpoint = resolve_checkpoint_path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f"checkpoint not found: {checkpoint}")
    if args.worker:
        with contextlib.redirect_stdout(sys.stderr):
            result = worker(args.worker, checkpoint)
        print(json.dumps(result))
        return
    results = {}
    for backend in ("python", "native"):
        env = os.environ.copy()
        env["GOMOKU_PATTERN_BACKEND"] = backend
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", backend,
                               "--checkpoint", str(checkpoint)], cwd=ROOT, env=env,
                              text=True, capture_output=True)
        if proc.returncode:
            raise RuntimeError(f"{backend} benchmark failed:\n{proc.stderr}")
        results[backend] = json.loads(proc.stdout)
    report = {"seed": SEED, "depth": 1, "simulations": 200, "rule_set": "normal",
              "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "machine": {"platform": platform.platform(), "processor": platform.processor(),
                          "architecture": platform.machine(), "logical_cpus": os.cpu_count(),
                          "python": sys.version}, "backends": results}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
