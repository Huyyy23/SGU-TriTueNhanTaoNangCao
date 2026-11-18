"""Reproducible self-play benchmarks and optional per-function profiling."""

from __future__ import annotations

import argparse
import cProfile
import json
import hashlib
import os
import pstats
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import psutil
import torch

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(values, p))


class ResourceMonitor:
    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.process = psutil.Process()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.rss_peak = 0
        self.cpu_util: list[float] = []
        self.gpu_util: list[float] = []
        self.gpu_mem: list[int] = []

    def _sample(self) -> None:
        while not self.stop_event.is_set():
            try:
                procs = [self.process, *self.process.children(recursive=True)]
                self.rss_peak = max(self.rss_peak, sum(p.memory_info().rss for p in procs if p.is_running()))
                cpu = sum(p.cpu_percent(interval=None) for p in procs if p.is_running())
                self.cpu_util.append(cpu)
            except (psutil.Error, OSError):
                pass
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1, check=True,
                )
                utilization, memory = result.stdout.splitlines()[0].split(",")
                self.gpu_util.append(float(utilization.strip()))
                self.gpu_mem.append(int(memory.strip()))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()


class FunctionProfiler:
    """Wall-clock inclusive/exclusive accounting for selected Python call boundaries."""

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, object]] = defaultdict(lambda: {"calls": 0, "inclusive_ns": 0, "exclusive_ns": 0, "samples_ns": []})
        self.callbacks: dict[str, list[int]] = defaultdict(list)
        self.local = threading.local()
        self.originals = []

    def patch(self, owner, name: str, label: str | None = None, callback_index: int | None = None) -> None:
        original = getattr(owner, name)
        key = label or f"{owner.__name__}.{name}"

        def measured(*args, **kwargs):
            stack = getattr(self.local, "stack", None)
            if stack is None:
                stack = self.local.stack = []
            frame = [time.perf_counter_ns(), 0]
            stack.append(frame)
            try:
                if callback_index is not None:
                    call_args = list(args)
                    callback = call_args[callback_index]

                    def timed_callback(*cb_args, **cb_kwargs):
                        callback_start = time.perf_counter_ns()
                        try:
                            return callback(*cb_args, **cb_kwargs)
                        finally:
                            callback_elapsed = time.perf_counter_ns() - callback_start
                            self.callbacks[f"{key}.callback"].append(callback_elapsed)
                            frame[1] += callback_elapsed

                    call_args[callback_index] = timed_callback
                    args = tuple(call_args)
                return original(*args, **kwargs)
            finally:
                elapsed = time.perf_counter_ns() - frame[0]
                stack.pop()
                frame_stats = self.stats[key]
                frame_stats["calls"] += 1
                frame_stats["inclusive_ns"] += elapsed
                frame_stats["exclusive_ns"] += elapsed - frame[1]
                frame_stats["samples_ns"].append(elapsed)
                if stack:
                    stack[-1][1] += elapsed

        setattr(owner, name, measured)
        self.originals.append((owner, name, original))

    def install(self) -> None:
        from az.mcts import MCTS, NativeMCTS
        from az.model import AZNet
        from gomoku.board import Board
        from gomoku.rules.normal import NormalRules

        for owner, names in ((MCTS, ("request", "receive", "_select_child", "_play_move", "_undo_move", "_get_planes", "_get_legal_mask")),
                             (NativeMCTS, ("request", "receive")),
                             (AZNet, ("predict",)),
                             (Board, ("play", "undo", "copy")),
                             (NormalRules, ("is_terminal", "is_win", "legal_moves"))):
            for name in names:
                self.patch(owner, name)
        self.patch(__import__("az.selfplay", fromlist=["board_to_planes"]), "board_to_planes")
        try:
            from gomoku_mcts_native import Tree
        except ImportError:
            return
        self.patch(Tree, "request", label="pybind.Tree.request", callback_index=2)
        for name in ("receive", "set_terminal"):
            self.patch(Tree, name, label=f"pybind.Tree.{name}")

    def restore(self) -> None:
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)

    def report(self) -> dict:
        out = {}
        for name, item in self.stats.items():
            samples = [int(value) for value in item["samples_ns"]]
            out[name] = {
                "calls": item["calls"],
                "inclusive_ms": item["inclusive_ns"] / 1e6,
                "exclusive_ms": item["exclusive_ns"] / 1e6,
                "p50_ms": percentile(samples, 50) / 1e6,
                "p95_ms": percentile(samples, 95) / 1e6,
            }
        for name, samples in self.callbacks.items():
            out[name] = {
                "calls": len(samples),
                "inclusive_ms": sum(samples) / 1e6,
                "p50_ms": percentile(samples, 50) / 1e6,
                "p95_ms": percentile(samples, 95) / 1e6,
                "note": "Python callback body time only; pybind call overhead is part of pybind.Tree.request",
            }
        return out


def load_runtime(checkpoint: str, device: str | None):
    from az.model import AZNet, load_checkpoint
    from config import load_config
    from gomoku.rules.normal import NormalRules

    cfg = load_config("configs/normal.yaml")
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if Path(checkpoint).is_file():
        model, _ = load_checkpoint(checkpoint, device=selected_device)
    else:
        model = AZNet(n_blocks=cfg["n_blocks"], n_filters=cfg["n_filters"]).to(selected_device)
    model.eval()
    return cfg, model, NormalRules(), selected_device


def run_one(model, rules, cfg, games: int, parallel: int, simulations: int, backend: str, seed: int,
            workers: int = 1, pool=None, profiling: FunctionProfiler | None = None) -> dict:
    from az.selfplay import run_games, run_games_parallel

    run_cfg = {**cfg, "parallel_games": parallel, "n_simulations": simulations,
               "mcts_backend": backend, "patterns_backend": "python"}
    inference_ms: list[float] = []
    leaves: list[int] = []
    original_predict = model.predict

    def timed_predict(features):
        started = time.perf_counter_ns()
        result = original_predict(features)
        inference_ms.append((time.perf_counter_ns() - started) / 1e6)
        leaves.append(int(len(features)))
        return result

    model.predict = timed_predict
    if profiling is not None:
        profiling.install()
    profiler = cProfile.Profile() if profiling is not None else None
    monitor = ResourceMonitor()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(selected_device := next(model.parameters()).device)
    start = time.perf_counter()
    try:
        with monitor:
            if profiler is not None:
                profiler.enable()
            if workers == 1:
                records = list(run_games(model, rules, run_cfg, games, seed=seed))
            else:
                records = run_games_parallel(pool, model, rules, run_cfg, games, seed=seed, workers=workers)
            if profiler is not None:
                profiler.disable()
            wall = time.perf_counter() - start
    finally:
        if profiler is not None:
            profiler.disable()
        model.predict = original_predict
        if profiling is not None:
            profiling.restore()
    moves = sum(record.moves_count for record in records)
    nn_calls = sum(record.nn_calls for record in records)
    simulations_done = simulations * moves
    result = {
        "seed": seed,
        "mcts_backend": backend, "parallel_games": parallel, "workers": workers,
        "games": len(records), "moves": moves, "wall_s": wall,
        "games_min": len(records) / wall * 60,
        "moves_s": moves / wall,
        "simulations": simulations_done, "simulations_s": simulations_done / wall,
        "nn_calls": nn_calls, "inference_batches": len(inference_ms) if workers == 1 else None,
        "inference_batch_mean": float(np.mean(leaves)) if leaves else None,
        "inference_batch_max": max(leaves) if leaves else None,
        "inference_latency_p50_ms": percentile(inference_ms, 50) if inference_ms else None,
        "inference_latency_p95_ms": percentile(inference_ms, 95) if inference_ms else None,
        "rss_peak_bytes_parent_and_children": monitor.rss_peak,
        "cpu_util_mean_pct_process_tree": float(np.mean(monitor.cpu_util)) if monitor.cpu_util else None,
        "gpu_util_mean_pct_sampled": float(np.mean(monitor.gpu_util)) if monitor.gpu_util else None,
        "gpu_memory_peak_mib_sampled": max(monitor.gpu_mem, default=None),
        "cuda_peak_allocated_bytes_parent": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        "cuda_peak_reserved_bytes_parent": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
        "record_mean_batch": float(np.mean([r.mean_batch for r in records])) if records else 0.0,
        "moves_per_game": [r.moves_count for r in records],
        "nn_calls_per_game": [r.nn_calls for r in records],
    }
    if profiler is not None:
        # cProfile is measured around the run separately from the direct timing wrappers.
        pstats_buffer = __import__("io").StringIO()
        pstats.Stats(profiler, stream=pstats_buffer).sort_stats("cumulative").print_stats(35)
        result["cprofile_top_cumulative"] = pstats_buffer.getvalue()
        result["function_profile"] = profiling.report()
    return result


def summarize_samples(samples_ns: list[int]) -> dict:
    return {
        "calls": len(samples_ns),
        "total_ms": sum(samples_ns) / 1e6,
        "p50_ms": percentile(samples_ns, 50) / 1e6,
        "p95_ms": percentile(samples_ns, 95) / 1e6,
    }


def profile_auxiliary_work() -> dict:
    """Measure work outside run_games: Renju rules, buffer copies, and durable logging."""
    import tempfile

    from az.game_log import GameLogger, GameRecord
    from az.replay_buffer import ReplayBuffer
    from gomoku.board import BLACK, WHITE, Board
    from gomoku.rules.renju import RenjuRules

    rng = np.random.default_rng(230001)
    renju = {"forbidden_cache_miss": [], "forbidden_cache_hit": [], "legal_moves": [], "is_win": []}
    for _ in range(24):
        board = Board()
        cells = rng.choice(121, size=12, replace=False)
        board.grid[cells[:6]] = BLACK
        board.grid[cells[6:]] = WHITE
        board.current_player = BLACK
        rules = RenjuRules()
        started = time.perf_counter_ns()
        rules.forbidden_moves(board)
        renju["forbidden_cache_miss"].append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        rules.forbidden_moves(board)
        renju["forbidden_cache_hit"].append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        rules.legal_moves(board)
        renju["legal_moves"].append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns()
        rules.is_win(board)
        renju["is_win"].append(time.perf_counter_ns() - started)

    buffer = ReplayBuffer(capacity=256)
    planes = np.zeros((4, 11, 11), dtype=np.uint8)
    policy = np.full(121, 1.0 / 121, dtype=np.float32)
    buffer_times = []
    for i in range(1000):
        started = time.perf_counter_ns()
        buffer.add(planes, policy, 0.0, "normal")
        buffer_times.append(time.perf_counter_ns() - started)

    logging_times = []
    with tempfile.TemporaryDirectory(prefix="gomoku-log-profile-") as tmp:
        logger = GameLogger(Path(tmp) / "games.jsonl")
        for game_id in range(48):
            record = GameRecord(
                game_id=game_id, timestamp="2026-11-18T00:00:00+00:00",
                rule_set="normal", first_player="selfplay", second_player="selfplay",
                winner="draw", first_player_won=False, end_reason="board_full",
                moves_count=121, duration_sec=1.0, config={"mcts": "python"},
            )
            started = time.perf_counter_ns()
            logger.log_game(record)
            logging_times.append(time.perf_counter_ns() - started)
    return {
        "renju_rules": {name: summarize_samples(samples) for name, samples in renju.items()},
        "replay_buffer_add": summarize_samples(buffer_times),
        "game_logger_fsync": summarize_samples(logging_times),
        "note": "Synthetic microbenchmarks; not included in self-play wall time.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("matrix", "workers", "profile"), required=True)
    parser.add_argument("--checkpoint", default="checkpoints/last.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker-games-per-worker", type=int, default=16)
    parser.add_argument("--worker-total-games", type=int, default=None,
                        help="Fixed total games per worker configuration; overrides per-worker scaling.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg, model, rules, device = load_runtime(args.checkpoint, args.device)
    cfg["n_simulations"] = args.simulations
    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if checkpoint_path.is_file() else None
    output = {"python": os.sys.version, "torch": torch.__version__, "device": str(device),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "logical_cpus": os.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total,
              "ram_available_before_bytes": psutil.virtual_memory().available,
              "checkpoint": str(checkpoint_path.resolve()), "checkpoint_sha256": checkpoint_sha256,
              "mode": args.mode, "simulations_per_move": args.simulations,
              "repeats": args.repeats, "results": [], "worker_groups": []}

    if args.mode == "matrix":
        # Warm each implementation with a true four-game batch before timed work.
        for backend in ("python", "cpp"):
            warm = run_one(model, rules, cfg, games=4, parallel=4, simulations=8,
                           backend=backend, seed=210000, workers=1)
            output.setdefault("warmup", []).append(warm)
        configs = [("python", 4), ("cpp", 16), ("cpp", 4), ("python", 16)]
        for rep in range(args.repeats):
            order = configs[rep % len(configs):] + configs[:rep % len(configs)]
            for backend, parallel in order:
                output["results"].append(run_one(model, rules, cfg, games=parallel, parallel=parallel,
                                                  simulations=args.simulations, backend=backend,
                                                  seed=20261118 + rep * 100 + parallel))
    elif args.mode == "workers":
        for workers in (1, 2, 4):
            games = (args.worker_total_games if args.worker_total_games is not None
                     else args.worker_games_per_worker * workers)
            parallel = min(int(cfg["parallel_games"]), games) if workers == 1 else (games + workers - 1) // workers
            group = {"workers": workers, "games": games, "parallel_games_per_process": parallel,
                     "warmup": [], "results": []}
            if workers == 1:
                for backend in ("python", "cpp"):
                    group["warmup"].append(run_one(
                        model, rules, cfg, games=games, parallel=parallel, simulations=8,
                        backend=backend, seed=219999, workers=1))
                for rep in range(args.repeats):
                    backends = ("python", "cpp") if rep % 2 == 0 else ("cpp", "python")
                    for backend in backends:
                        group["results"].append(run_one(
                            model, rules, cfg, games=games, parallel=parallel,
                            simulations=args.simulations, backend=backend,
                            seed=220000 + rep * 10, workers=1))
            else:
                startup_started = time.perf_counter()
                with ProcessPoolExecutor(max_workers=workers) as pool:
                    for backend in ("python", "cpp"):
                        group["warmup"].append(run_one(
                            model, rules, cfg, games=games, parallel=parallel, simulations=8,
                            backend=backend, seed=219999, workers=workers, pool=pool))
                    group["worker_startup_plus_warmup_sec"] = time.perf_counter() - startup_started
                    for rep in range(args.repeats):
                        backends = ("python", "cpp") if rep % 2 == 0 else ("cpp", "python")
                        for backend in backends:
                            group["results"].append(run_one(
                                model, rules, cfg, games=games, parallel=parallel,
                                simulations=args.simulations, backend=backend,
                                seed=220000 + rep * 10, workers=workers, pool=pool))
            output["worker_groups"].append(group)
    else:
        for backend in ("python", "cpp"):
            prof = FunctionProfiler()
            result = run_one(model, rules, cfg, games=4, parallel=4, simulations=args.simulations,
                             backend=backend, seed=230000, profiling=prof)
            output["results"].append(result)
        output["auxiliary_microbenchmarks"] = profile_auxiliary_work()

    output["ram_available_after_bytes"] = psutil.virtual_memory().available
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"out": args.out, "mode": args.mode, "results": output["results"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
