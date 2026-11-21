from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from numbers import Integral
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from az.evaluator import play_game
from az.logging_v2 import (
    TOURNAMENT_DB,
    connect as connect_tournament_db,
    initialize as initialize_tournament_db,
    record_tournament_experiment,
    record_tournament_game,
    tournament_agent_id,
    write_event,
)
from config import load_config
from gomoku.rules import get_rules
from players.factory import AGENT_CHOICES, create_player
import itertools
import platform
import random
import subprocess
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Callable
import numpy as np
from az.evaluator import wilson_interval
from gomoku.board import BLACK, EMPTY, SIZE
from gomoku.runner import Runner
from players.base import IllegalMoveError
AZ_KINDS = ("az", "az_v0", "az_v1_tactical", "az_v1_threat")
KNOWN_KINDS = ("random", "heuristic", "alphabeta", "mcts-pure", "az_renju") + AZ_KINDS
SPEC_KEYS = {"sims", "depth", "pruning", "checkpoint"}
@dataclass(frozen=True)
class Budget:
    mode: str  # "sims" or "time"
    value: float

    @classmethod
    def parse(cls, text: str) -> "Budget":
        mode, sep, raw = str(text).partition(":")
        mode = mode.strip().lower()
        if not sep or mode not in ("sims", "time"):
            raise ValueError(f"budget {text!r} must look like sims:N or time:S")
        try:
            value = int(raw) if mode == "sims" else float(raw)
        except ValueError:
            raise ValueError(f"budget {text!r}: {raw!r} is not a number") from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"budget {text!r} must be positive")
        return cls(mode, value)

    @property
    def label(self) -> str:
        return f"{self.mode}{self.value:g}"

    def as_dict(self) -> dict:
        return {"mode": self.mode, "value": self.value, "text": f"{self.mode}:{self.value:g}"}


# -- agent specs ---------------------------------------------------------------


@dataclass(frozen=True)
class AgentSpec:
    """One entry of --agents: `kind` or `kind@key=value@key=value`."""

    label: str
    kind: str
    params: dict = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def parse(cls, text: str) -> "AgentSpec":
        text = text.strip()
        if not text:
            raise ValueError("empty agent name in --agents")
        head, *options = text.split("@")
        kind = ALIASES.get(head.strip(), head.strip())
        if kind not in KNOWN_KINDS:
            choices = ", ".join(sorted(set(KNOWN_KINDS) | set(ALIASES)))
            raise ValueError(f"unknown agent {head!r}; choose from {choices}")
        params: dict[str, Any] = {}
        for option in options:
            key, sep, value = option.partition("=")
            key = key.strip()
            if not sep or key not in SPEC_KEYS:
                raise ValueError(f"agent {text!r}: option {option!r} must be one of {sorted(SPEC_KEYS)}=value")
            if key in ("sims", "depth"):
                try:
                    params[key] = int(value)
                except ValueError:
                    raise ValueError(f"agent {text!r}: {key} must be an integer") from None
                if params[key] <= 0:
                    raise ValueError(f"agent {text!r}: {key} must be positive")
            elif key == "pruning":
                if value.lower() not in ("true", "false", "1", "0", "on", "off"):
                    raise ValueError(f"agent {text!r}: pruning must be true or false")
                params[key] = value.lower() in ("true", "1", "on")
            else:
                params[key] = value
        if "depth" in params or "pruning" in params:
            if kind != "alphabeta":
                raise ValueError(f"agent {text!r}: depth/pruning only apply to alphabeta")
        if "sims" in params and kind not in ("mcts-pure",) + AZ_KINDS:
            raise ValueError(f"agent {text!r}: sims only applies to mcts_pure and az_*")
        if "checkpoint" in params and kind not in AZ_KINDS:
            raise ValueError(f"agent {text!r}: checkpoint only applies to az_*")
        return cls(label=text, kind=kind, params=params)


def parse_agents(text: str) -> list[AgentSpec]:
    specs = [AgentSpec.parse(part) for part in str(text).split(",") if part.strip()]
    if len(specs) < 2:
        raise ValueError("--agents needs at least two agents")
    labels = [spec.label for spec in specs]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"--agents lists {duplicates} more than once")
    return specs


@dataclass
class Defaults:
    checkpoint: str | None = None
    config: str | None = None
    alphabeta_depth: int = 1
    alphabeta_pruning: bool = True
    mcts_rollouts_time_mode: int = 200
    time_sims_cap: int = 100_000


def agent_settings(spec: AgentSpec, budget: Budget, defaults: Defaults) -> dict:
    """Resolve one agent into factory arguments plus a plain-language budget note."""
    kind, params = spec.kind, spec.params
    factory: dict[str, Any] = {"agent": kind}
    if kind in AZ_KINDS:
        checkpoint = params.get("checkpoint", defaults.checkpoint)
        if not checkpoint:
            raise ValueError(f"agent {spec.label!r} needs --checkpoint (or @checkpoint=PATH)")
        if not Path(checkpoint).is_file():
            raise ValueError(f"agent {spec.label!r}: checkpoint {checkpoint!r} not found")
        factory["checkpoint"] = str(checkpoint)
        factory["config"] = defaults.config
        if budget.mode == "sims":
            factory["n_simulations"] = int(params.get("sims", budget.value))
            factory["time_budget_sec"] = None
            note = f"{factory['n_simulations']} PUCT simulations per move"
        else:
            factory["n_simulations"] = int(params.get("sims", defaults.time_sims_cap))
            factory["time_budget_sec"] = float(budget.value)
            note = (f"stops after {budget.value:g} s per move "
                    f"(simulations capped at {factory['n_simulations']})")
        return {"factory": factory, "budget_applies": True, "budget_enforced": True, "budget_note": note}
    if kind == "mcts-pure":
        if budget.mode == "sims":
            factory["rollouts"] = int(params.get("sims", budget.value))
            return {"factory": factory, "budget_applies": True, "budget_enforced": True,
                    "budget_note": f"{factory['rollouts']} UCT rollouts per move"}
        factory["rollouts"] = int(params.get("sims", defaults.mcts_rollouts_time_mode))
        factory["time_budget_sec"] = float(budget.value)
        return {"factory": factory, "budget_applies": True, "budget_enforced": True,
                "budget_note": f"soft deadline {budget.value:g} s per move; rollout blocks may finish after it"}
    if kind == "alphabeta":
        factory["depth"] = int(params.get("depth", defaults.alphabeta_depth))
        factory["pruning"] = bool(params.get("pruning", defaults.alphabeta_pruning))
        if budget.mode == "sims":
            note = f"fixed depth {factory['depth']} (pruning {'on' if factory['pruning'] else 'off'}); sims:N does not apply"
        else:
            factory["time_budget_sec"] = float(budget.value)
            note = f"iterative deepening with soft deadline {budget.value:g} s per move"
        return {"factory": factory, "budget_applies": budget.mode == "time", "budget_enforced": budget.mode == "time",
                "budget_note": note}
    # random, heuristic: no search at all.
    return {"factory": factory, "budget_applies": False, "budget_enforced": False,
            "budget_note": "no search; the budget does not apply"}


def default_builder(settings: dict, rules, seed: int):
    """Build a player through the shared factory used by scripts/play.py and the UI."""
    from players.factory import create_player

    kwargs = dict(settings["factory"])
    agent = kwargs.pop("agent")
    if settings.get("backend"):
        kwargs["mcts_backend"] = settings["backend"]["mcts"]
        kwargs["patterns_backend"] = settings["backend"]["patterns"]
    return create_player(agent, rules, seed=seed, **kwargs)

# -- instrumentation -----------------------------------------------------------


class Probe:
    """Wrap a player: time every move and read its search counters without changing it."""

    def __init__(self, player, label: str, opening: list[int] | None = None,
                 collect_telemetry: bool = True) -> None:
        self.player = player
        self.name = label
        self.rules = getattr(player, "rules", None)
        self.opening = opening or []
        self.moves: list[dict] = []
        self.collect_telemetry = collect_telemetry
        self._nn_calls = 0
        self._base_time = 0.0
        self._threat: dict | None = None
        if not collect_telemetry:
            return

        base = getattr(player, "base", None)
        owner = base if base is not None else player
        model = getattr(owner, "model", None)
        if model is not None and hasattr(model, "predict"):
            predict = model.predict

            def counted_predict(*args, **kwargs):
                self._nn_calls += 1
                return predict(*args, **kwargs)

            model.predict = counted_predict
        if base is not None:
            base_select = base.select_move

            def timed_base(board):
                started = time.perf_counter()
                try:
                    return base_select(board)
                finally:
                    self._base_time += time.perf_counter() - started

            base.select_move = timed_base
        searcher = getattr(player, "searcher", None)
        if searcher is not None and getattr(player, "threat_enabled", False):
            search = searcher.search

            def recorded_search(*args, **kwargs):
                started = time.perf_counter()
                result = search(*args, **kwargs)
                self._threat = {
                    "nodes": int(result.nodes),
                    "triggered": bool(result.triggered),
                    "limit": bool(result.limit_reached),
                    "ms": round((time.perf_counter() - started) * 1000, 3),
                }
                return result

            searcher.search = recorded_search

    def select_move(self, board) -> tuple[int, int]:
        ply = len(board.history)
        if ply < len(self.opening):
            return divmod(self.opening[ply], SIZE)  # shared random opening, not timed
        if not self.collect_telemetry:
            move = self.player.select_move(board)
            if not isinstance(move, tuple) or len(move) != 2 or not all(isinstance(value, Integral) for value in move):
                raise IllegalMoveError(f"{self.name} returned malformed move {move!r}")
            return move
        self._nn_calls, self._base_time, self._threat = 0, 0.0, None
        started = time.perf_counter()
        move = self.player.select_move(board)
        elapsed = time.perf_counter() - started
        if not isinstance(move, tuple) or len(move) != 2 or not all(isinstance(value, Integral) for value in move):
            raise IllegalMoveError(f"{self.name} returned malformed move {move!r}")
        record: dict[str, Any] = {"t": elapsed}
        if getattr(self.player, "nodes_visited", None) is not None:
            record["nodes"] = int(self.player.nodes_visited)
        if getattr(self.player, "simulations_run", None) is not None:
            record["sims"] = int(self.player.simulations_run)
        if getattr(self.player, "nn_calls", None) is not None:
            record["nn_calls"] = int(self.player.nn_calls)
        if hasattr(self.player, "last_override_applied"):
            record["override"] = self.player.last_override_reason if self.player.last_override_applied else None
            record["extra_t"] = max(0.0, elapsed - self._base_time)
        if self._threat is not None:
            record["threat"] = self._threat
        self.moves.append(record)
        record["raw"] = {"move_index": ply, "player": int(board.current_player),
                         "row": int(move[0]), "col": int(move[1]), "latency_sec": elapsed,
                         "nodes_visited": record.get("nodes"), "simulations_run": record.get("sims"),
                         "nn_calls": record.get("nn_calls"), "override_applied": bool(record.get("override")),
                         "override_reason": record.get("override")}
        return move

def _move_stats(moves: list[dict], time_budget: float | None) -> dict:
    times = [m["t"] for m in moves]
    stats: dict[str, Any] = {"moves": len(moves), "time_sum": sum(times), "times": times}
    if time_budget is not None:
        stats["over_budget"] = sum(1 for t in times if t > time_budget)
    for key in ("nodes", "sims", "nn_calls"):
        values = [m.get(key, 0) for m in moves]
        if any(key in m for m in moves):
            stats[key] = values
    if any("override" in m for m in moves):
        stats["overrides"] = dict(Counter(m["override"] for m in moves if m.get("override")))
        stats["extra_t_sum"] = sum(m.get("extra_t", 0.0) for m in moves)
    threat = [m["threat"] for m in moves if "threat" in m]
    if threat:
        stats["threat"] = {
            "searches": len(threat),
            "activations": sum(1 for t in threat if t["triggered"]),
            "limit_hits": sum(1 for t in threat if t["limit"]),
            "nodes": [t["nodes"] for t in threat],
            "ms": [t["ms"] for t in threat],
        }
    return stats


# -- one game ------------------------------------------------------------------


def derive_seed(*parts) -> int:
    """Stable across processes and platforms (unlike hash())."""
    return zlib.crc32("|".join(map(str, parts)).encode("utf-8")) & 0x7FFFFFFF


def random_opening(rules_name: str, n_moves: int, seed: int) -> list[int]:
    """Uniform random legal opening of n_moves plies that does not end the game."""
    if n_moves <= 0:
        return []
    from gomoku.board import Board

    rules = get_rules(rules_name)
    rng = random.Random(seed)
    for _ in range(100):
        board, moves = Board(), []
        for _ in range(n_moves):
            move = rng.choice(rules.legal_moves(board))
            board.play(*divmod(move, SIZE))
            moves.append(move)
            if rules.is_terminal(board):
                break
        else:
            return moves
    raise RuntimeError("could not draw a non-terminal random opening")


def _set_torch_threads(n: int | None) -> None:
    """One-position inference is ~10x slower with torch's default thread pool on CPU."""
    if not n:
        return
    try:
        import torch
    except ImportError:
        return
    if torch.get_num_threads() != n:
        torch.set_num_threads(n)


def play_game_task(task: dict, builder: Callable | None = None) -> dict:
    """Play one scheduled game and return its JSON record (top-level so workers can pickle it)."""
    from az.backend import configure_backends
    backends = task.get("backends")
    if backends is None:
        backends = configure_backends("python", allow_build=False)
    configure_backends("python", mcts_backend=backends["mcts"], patterns_backend=backends["patterns"],
                       allow_build=False)
    builder = builder or default_builder
    rules = get_rules(task["ruleset"])
    opening = task["opening"]
    black_label, white_label = task["black"], task["white"]
    collect_telemetry = task.get("collect_telemetry", True)
    black = Probe(builder(task["settings"][black_label], rules, task["seeds"]["black"]), black_label, opening,
                  collect_telemetry)
    white = Probe(builder(task["settings"][white_label], rules, task["seeds"]["white"]), white_label, opening,
                  collect_telemetry)

    started = time.perf_counter()
    moves: list[int] = []

    def on_move(board, _player, _side):
        moves.append(int(board.history[-1]))

    illegal_by, error = None, None
    try:
        result = Runner(rules).play(black, white, on_move=on_move)
        if result.winner == EMPTY:
            winner, end_reason = "draw", "board_full"
        else:
            winner, end_reason = ("black" if result.winner == BLACK else "white"), "five_in_row"
    except IllegalMoveError as exc:
        # The side to move after `moves` returned the illegal move: it loses the game.
        offender = "black" if len(moves) % 2 == 0 else "white"
        illegal_by = black_label if offender == "black" else white_label
        winner = "white" if offender == "black" else "black"
        end_reason, error = "illegal_move", str(exc)

    winner_agent = {"black": black_label, "white": white_label}.get(winner)
    budget_time = task.get("time_budget")
    return {
        "run_id": task["run_id"],
        "game_id": task["game_id"],
        "pair": [task["agent_a"], task["agent_b"]],
        "pair_game": task["pair_game"],
        "first_player": black_label,
        "second_player": white_label,
        "rule_set": task["ruleset"],
        "winner": winner,
        "winner_agent": winner_agent,
        "first_player_won": winner == "black",
        "end_reason": end_reason,
        "illegal_by": illegal_by,
        "error": error,
        "moves_count": len(moves),
        "moves": moves,
        **({"raw_moves": sorted([m["raw"] for m in black.moves + white.moves], key=lambda m: m["move_index"])}
           if collect_telemetry else {}),
        "opening_plies": len(opening),
        "duration_sec": round(time.perf_counter() - started, 4),
        "seeds": task["seeds"],
        "stats": {
            "black": _move_stats(black.moves, budget_time),
            "white": _move_stats(white.moves, budget_time),
        },
        "backends": backends,
        "components_used": {
            "mcts": backends["mcts"] if any(task["settings"][label]["factory"]["agent"] in AZ_KINDS for label in (black_label, white_label)) else "not_used",
            "patterns": backends["patterns"] if any(task["settings"][label]["factory"]["agent"] in {"alphabeta", "heuristic", "az_v1_threat"} for label in (black_label, white_label)) else "not_used",
        },
    }


# -- schedule and aggregation --------------------------------------------------


def schedule(labels: list[str], games: int, seed: int, rules_name: str, opening_plies: int) -> list[dict]:
    """All pairs, `games` per pair. Even pair_game: A is Black; odd: B is Black.

    Games 2k and 2k+1 share the opening and the agents' seeds, so each colour-swapped
    pair of games differs only in who moves first.
    """
    if games < 2 or games % 2:
        raise ValueError("--games must be an even number >= 2 so colours split evenly")
    tasks = []
    for pair_index, (a, b) in enumerate(itertools.combinations(labels, 2)):
        for pair_game in range(games):
            block = pair_game // 2
            seed_a = derive_seed(seed, a, b, block, "A")
            seed_b = derive_seed(seed, a, b, block, "B")
            opening_seed = derive_seed(seed, a, b, block, "opening")
            a_black = pair_game % 2 == 0
            tasks.append({
                "game_id": len(tasks),
                "pair_index": pair_index,
                "agent_a": a,
                "agent_b": b,
                "pair_game": pair_game,
                "black": a if a_black else b,
                "white": b if a_black else a,
                "seeds": {
                    "black": seed_a if a_black else seed_b,
                    "white": seed_b if a_black else seed_a,
                    "opening": opening_seed if opening_plies else None,
                },
                "opening": random_opening(rules_name, opening_plies, opening_seed),
                "ruleset": rules_name,
            })
    return tasks


def _percentile(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 6) if values else None


def _mean(values) -> float | None:
    values = list(values)
    return round(float(np.mean(values)), 6) if values else None


class _AgentAcc:
    """Accumulate one agent's move statistics over many games."""

    def __init__(self) -> None:
        self.times: list[float] = []
        self.nodes: list[int] = []
        self.sims: list[int] = []
        self.nn_calls: list[int] = []
        self.over_budget = 0
        self.overrides: Counter = Counter()
        self.extra_t = 0.0
        self.has_override = False
        self.threat = {"searches": 0, "activations": 0, "limit_hits": 0, "nodes": [], "ms": []}
        self.has_threat = False
        self.games = 0

    def add(self, stats: dict) -> None:
        self.games += 1
        self.times += stats["times"]
        self.nodes += stats.get("nodes", [])
        self.sims += stats.get("sims", [])
        self.nn_calls += stats.get("nn_calls", [])
        self.over_budget += stats.get("over_budget", 0)
        if "overrides" in stats:
            self.has_override = True
            self.overrides.update(stats["overrides"])
            self.extra_t += stats.get("extra_t_sum", 0.0)
        if "threat" in stats:
            self.has_threat = True
            for key in ("searches", "activations", "limit_hits"):
                self.threat[key] += stats["threat"][key]
            self.threat["nodes"] += stats["threat"]["nodes"]
            self.threat["ms"] += stats["threat"]["ms"]

    def report(self, time_budget: float | None) -> dict:
        moves = len(self.times)
        out: dict[str, Any] = {
            "moves_timed": moves,
            "mean_time_per_move_sec": _mean(self.times),
            "p95_time_per_move_sec": _percentile(self.times, 95),
            "max_time_per_move_sec": round(max(self.times), 6) if self.times else None,
        }
        if time_budget is not None:
            out["moves_over_time_budget"] = self.over_budget
        if self.nodes:
            out["mean_nodes_per_move"] = _mean(self.nodes)
            out["p95_nodes_per_move"] = _percentile(self.nodes, 95)
        if self.sims:
            out["rollouts_per_move"] = _mean(self.sims)
        if self.nn_calls or self.has_override:
            out["mean_nn_calls_per_move"] = round(sum(self.nn_calls) / moves, 3) if moves else None
            out["mean_nn_calls_per_game"] = round(sum(self.nn_calls) / self.games, 3) if self.games else None
        if self.has_override:
            total = sum(self.overrides.values())
            out["override_count"] = total
            out["override_rate"] = round(total / moves, 4) if moves else None
            out["override_reasons"] = dict(self.overrides)
            out["mean_extra_time_per_move_sec"] = round(self.extra_t / moves, 6) if moves else None
        if self.has_threat:
            t = self.threat
            out["threat_space"] = {
                "searches": t["searches"],
                "activations": t["activations"],
                "limit_hits": t["limit_hits"],
                "mean_nodes": _mean(t["nodes"]),
                "max_nodes": max(t["nodes"]) if t["nodes"] else None,
                "mean_ms": _mean(t["ms"]),
                "p95_ms": _percentile(t["ms"], 95),
            }
        return out


def _score(wins: int, draws: int, games: int) -> float | None:
    return round((wins + 0.5 * draws) / games, 4) if games else None


def aggregate(records: list[dict], labels: list[str], time_budget: float | None) -> dict:
    """Pair table, per-agent table and matrix from game records (pure function, easy to test)."""
    pairs = []
    agent_acc = {label: _AgentAcc() for label in labels}
    agent_wdl = {label: Counter() for label in labels}
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        by_pair.setdefault(tuple(record["pair"]), []).append(record)

    for a, b in itertools.combinations(labels, 2):
        games = sorted(by_pair.get((a, b), []), key=lambda r: r["pair_game"])
        acc = {a: _AgentAcc(), b: _AgentAcc()}
        wdl = {a: Counter(), b: Counter()}
        sequences = {a: set(), b: set()}  # distinct games with that agent as Black
        lengths, illegal = [], Counter()
        for record in games:
            for colour, label in (("black", record["first_player"]), ("white", record["second_player"])):
                acc[label].add(record["stats"][colour])
                agent_acc[label].add(record["stats"][colour])
                if record["winner"] == "draw":
                    outcome = "D"
                else:
                    outcome = "W" if record["winner"] == colour else "L"
                wdl[label][outcome] += 1
                wdl[label][f"{colour}_{outcome}"] += 1
                wdl[label][f"{colour}_games"] += 1
                agent_wdl[label][outcome] += 1
                agent_wdl[label][f"{colour}_{outcome}"] += 1
                agent_wdl[label][f"{colour}_games"] += 1
            sequences[record["first_player"]].add(tuple(record["moves"]))
            lengths.append(record["moves_count"])
            if record["illegal_by"]:
                illegal[record["illegal_by"]] += 1
        n = len(games)
        row: dict[str, Any] = {"agent_a": a, "agent_b": b, "games": n}
        for side, label in (("a", a), ("b", b)):
            c = wdl[label]
            row[f"wins_{side}"] = c["W"]
            row[f"losses_{side}"] = c["L"]
            row[f"score_{side}"] = _score(c["W"], c["D"], n)
            row[f"wilson95_{side}"] = list(wilson_interval(c["W"] + 0.5 * c["D"], n)) if n else None
            row[f"games_{side}_black"] = c["black_games"]
            row[f"games_{side}_white"] = c["white_games"]
            row[f"score_{side}_black"] = _score(c["black_W"], c["black_D"], c["black_games"])
            row[f"score_{side}_white"] = _score(c["white_W"], c["white_D"], c["white_games"])
            row[f"distinct_games_{side}_black"] = len(sequences[label])
            row[f"illegal_moves_{side}"] = illegal[label]
            row[f"stats_{side}"] = acc[label].report(time_budget)
        row["draws"] = wdl[a]["D"]
        row["mean_moves_per_game"] = _mean(lengths)
        # T31 asks for the seed of every pair: keep each game's seeds so one game can be replayed.
        row["game_seeds"] = [
            {"pair_game": r["pair_game"], "black_agent": r["first_player"], "white_agent": r["second_player"],
             "seeds": r.get("seeds")}
            for r in games
        ]
        pairs.append(row)

    agents = {}
    for label in labels:
        c = agent_wdl[label]
        n = c["W"] + c["D"] + c["L"]
        agents[label] = {
            "games": n,
            "wins": c["W"], "draws": c["D"], "losses": c["L"],
            "score": _score(c["W"], c["D"], n),
            "wilson95": list(wilson_interval(c["W"] + 0.5 * c["D"], n)) if n else None,
            "games_black": c["black_games"],
            "games_white": c["white_games"],
            "score_black": _score(c["black_W"], c["black_D"], c["black_games"]),
            "score_white": _score(c["white_W"], c["white_D"], c["white_games"]),
            **agent_acc[label].report(time_budget),
        }

    matrix = {row: {col: None for col in labels if col != row} for row in labels}
    for row in pairs:
        matrix[row["agent_a"]][row["agent_b"]] = row["score_a"]
        matrix[row["agent_b"]][row["agent_a"]] = row["score_b"]
    return {"pairs": pairs, "agents": agents, "matrix": matrix}



def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": run("git", "rev-parse", "HEAD") or None,
            "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no"))}


def _versions() -> dict:
    versions = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()}
    try:
        import torch

        versions["torch"] = torch.__version__
    except ImportError:
        versions["torch"] = None
    return versions


# -- run -----------------------------------------------------------------------


def run_tournament(
    entries: dict[str, dict],
    *,
    games: int,
    budget: Budget,
    ruleset: str = "normal",
    seed: int = 0,
    run_id: str | None = None,
    workers: int = 1,
    opening_plies: int = 0,
    builder: Callable | None = None,
    torch_threads: int | None = 1,
    extra_config: dict | None = None,
    log: Callable[[str], None] | None = None,
    resume: bool = False,
    pair: str | None = None,
    seed_scheme: str = "paired-crc32",
    backends: dict | None = None,
    db_path: str | Path = TOURNAMENT_DB,
    logging_level: str = "standard",
) -> tuple[Path, dict]:
    """Play a round-robin; NONE returns in-memory results without durable output."""
    if logging_level not in {"none", "standard"}:
        raise ValueError("logging_level must be 'none' or 'standard'")
    if resume and logging_level == "none":
        raise ValueError("resume requires STANDARD logging")
    if log is None:
        log = (lambda message: print(message, flush=True)) if logging_level == "standard" else (lambda _message: None)
    kinds = {settings["factory"]["agent"] for settings in entries.values()}
    if backends is None:
        from az.backend import configure_backends
        backends = configure_backends(
            "auto",
            mcts_backend=None if kinds & set(AZ_KINDS) else "python",
            patterns_backend=None if kinds & {"alphabeta", "heuristic", "az_v1_threat"} else "python",
        )
    components_used = {
        "mcts": backends["mcts"] if kinds & set(AZ_KINDS) else "not_used",
        "patterns": backends["patterns"] if kinds & {"alphabeta", "heuristic", "az_v1_threat"} else "not_used",
    }
    entries = {label: {**settings, "backend": backends} for label, settings in entries.items()}
    labels = list(entries)
    if games < 2 or games % 2 or len(labels) < 2:
        raise ValueError("at least two agents and a positive even game count required")
    if seed_scheme not in {"paired-crc32", "legacy-v1"}:
        raise ValueError("unknown seed scheme")
    get_rules(ruleset)  # fail fast on an unknown rule set
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if workers > 1 and builder is not None:
        raise ValueError("a custom builder only works with workers=1")
    if opening_plies < 0:
        raise ValueError("--random-opening must be >= 0")
    if resume and not run_id:
        raise ValueError("--resume requires an explicit --run-id")
    run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}_{ruleset}_{budget.label}_s{seed}"
    tasks = schedule(labels, games, seed, ruleset, opening_plies)
    if seed_scheme == "legacy-v1":
        if opening_plies:
            raise ValueError("legacy seed scheme does not support random openings")
        for task in tasks:
            base = seed + task["pair_index"] * 100000 + task["pair_game"] * 2
            task["seeds"] = {"black": base if task["pair_game"] % 2 == 0 else base + 1,
                             "white": base + 1 if task["pair_game"] % 2 == 0 else base, "opening": None}
    if pair:
        selected = pair.split(",")
        if len(selected) != 2 or len(set(selected)) != 2 or not set(selected) <= set(labels):
            raise ValueError("--pair must contain two distinct names from --agents")
        tasks = [task for task in tasks if {task["agent_a"], task["agent_b"]} == set(selected)]
        for index, task in enumerate(tasks):
            task["game_id"] = index
    time_budget = budget.value if budget.mode == "time" else None
    from az.model import resolve_checkpoint_path
    entries = {
        label: {**settings, "factory": {
            **settings["factory"],
            **({"checkpoint": str(resolve_checkpoint_path(settings["factory"]["checkpoint"]))}
               if settings["factory"].get("checkpoint") else {}),
        }}
        for label, settings in entries.items()
    }
    checkpoints = sorted({s["factory"]["checkpoint"] for s in entries.values() if s["factory"].get("checkpoint")})
    hashes_before = {path: _sha256(path) for path in checkpoints}

    _set_torch_threads(torch_threads)

    # Build every agent once before the first game: a bad checkpoint, or a YAML that does not match
    # the checkpoint, fails here with a clear message, not hours into the run.
    preflight_rules = get_rules(ruleset)
    for label in sorted({task["agent_a"] for task in tasks} | {task["agent_b"] for task in tasks}):
        try:
            (builder or default_builder)(entries[label], preflight_rules, seed)
        except Exception as exc:
            raise ValueError(f"agent {label!r} could not be created: {type(exc).__name__}: {exc}") from exc

    config = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([Path(sys.argv[0]).name] + sys.argv[1:]),
        "agents": labels,
        "agent_settings": entries,
        "games_per_pair": games,
        "pairs": len(labels) * (len(labels) - 1) // 2,
        "total_games": len(tasks),
        "budget": budget.as_dict(),
        "ruleset": ruleset,
        "seed": seed,
        "seed_scheme": seed_scheme,
        "pair": pair,
        "algorithm_version": "integrated-t35-t31-v2",
        "seed_rule": "crc32 paired colour-swap seeds" if seed_scheme == "paired-crc32" else "seed + pair_index*100000 + pair_game*2; opponent +1",
        "random_opening_plies": opening_plies,
        "score_rule": "win = 1, draw = 0.5, loss = 0; Wilson 95% interval on (wins + 0.5 * draws) / games",
        "illegal_move_rule": "an illegal move loses the game and is counted in illegal_moves_*",
        "workers": workers,
        "torch_threads_per_game": torch_threads,
        "checkpoint_sha256": hashes_before,
        "git": _git_state() if logging_level == "standard" else None,
        "versions": _versions() if logging_level == "standard" else None,
        **(extra_config or {}),
        "backends": backends,
        "components_used": components_used,
        "logging_level": logging_level,
    }
    database_config = config
    database_metadata = {"pair": pair}
    db_path = Path(db_path)
    agent_ids = {}
    if logging_level == "standard":
        initialize_tournament_db(db_path, "tournament")
        with connect_tournament_db(db_path) as connection:
            saved = connection.execute(
                "SELECT started_at, backend, rule_set, seed, config_json, metadata_json "
                "FROM experiments WHERE experiment_id=?", (run_id,)
            ).fetchone()
            if saved is not None and not resume:
                raise FileExistsError(f"experiment {run_id!r} already exists in {db_path}")
            if resume:
                if saved is None:
                    raise ValueError(f"resume requires an existing SQLite experiment: {run_id}")
                database_config = json.loads(saved["config_json"])
                database_metadata = json.loads(saved["metadata_json"])
                resume_fields = (
                    "agents", "agent_settings", "games_per_pair", "budget", "ruleset", "seed", "seed_scheme",
                    "pair", "random_opening_plies", "checkpoint_sha256", "torch_threads_per_game",
                    "algorithm_version", "loaded_config", "backends", "components_used",
                )
                if any(database_config.get(key) != config.get(key) for key in resume_fields):
                    raise ValueError("resume configuration or checkpoint does not match saved SQLite experiment")
                config = database_config
                database_metadata = json.loads(saved["metadata_json"])
            record_tournament_experiment(
                connection, experiment_id=run_id, started_at=database_config["created_at"],
                backend=backends.get("requested_backend", "auto"), rule_set=ruleset, seed=seed,
                config=database_config, metadata=database_metadata,
            )
            for label, settings in entries.items():
                factory = settings["factory"]
                agent_ids[label] = tournament_agent_id(
                    label=label, algorithm=factory["agent"],
                    checkpoint_sha256=hashes_before.get(factory.get("checkpoint")),
                    backend=json.dumps(backends, sort_keys=True), rule_set=ruleset, config=settings,
                )
                from az.logging_v2 import record_tournament_agent, record_tournament_match
                record_tournament_agent(
                    connection, agent_id=agent_ids[label], label=label, algorithm=factory["agent"],
                    checkpoint_sha256=hashes_before.get(factory.get("checkpoint")),
                    backend=json.dumps(backends, sort_keys=True), rule_set=ruleset, config=settings,
                )
            for task in tasks:
                pair_id = f"{task['agent_a']}|{task['agent_b']}:{task['pair_game'] // 2}"
                record_tournament_match(
                    connection, match_id=f"{run_id}:{pair_id}", experiment_id=run_id,
                    agent_a_id=agent_ids[task["agent_a"]], agent_b_id=agent_ids[task["agent_b"]],
                    pair_id=pair_id, rule_set=ruleset, seed=task["seeds"]["opening"], metadata={"games": 2},
                )

    def persist_game(record: dict) -> None:
        if logging_level != "standard":
            return
        with connect_tournament_db(db_path) as connection:
            pair_id = f"{record['pair'][0]}|{record['pair'][1]}:{record['pair_game'] // 2}"
            record_tournament_game(
                connection,
                experiment_id=run_id,
                local_game_id=record["game_id"],
                match_id=f"{run_id}:{pair_id}",
                black_agent_id=agent_ids[record["first_player"]],
                white_agent_id=agent_ids[record["second_player"]],
                record=record,
            )
    records = []
    if resume:
        with connect_tournament_db(db_path) as connection:
            records = [
                json.loads(row["payload_json"])
                for row in connection.execute(
                    "SELECT payload_json FROM games WHERE experiment_id=? ORDER BY CAST(local_game_id AS INTEGER)",
                    (run_id,),
                )
            ]
        expected = {task["game_id"]: task for task in tasks}
        seen = set()
        for record in records:
            task = expected.get(record["game_id"])
            if task is None or record["game_id"] in seen:
                raise ValueError("invalid or duplicate resume game ID")
            for key in ("pair_game", "seeds", "opening_plies", "rule_set", "first_player", "second_player", "pair"):
                value = {"opening_plies": len(task["opening"]), "rule_set": ruleset,
                         "first_player": task["black"], "second_player": task["white"],
                         "pair": [task["agent_a"], task["agent_b"]]}.get(key, task.get(key))
                if record.get(key) != value:
                    raise ValueError(f"invalid resume game identity or seed: {key}")
            if record.get("winner") not in {"black", "white", "draw"} or len(record.get("moves", [])) != record["moves_count"]:
                raise ValueError("invalid resume result or moves")
            seen.add(record["game_id"])
        for record in records:
            persist_game(record)
        tasks = [task for task in tasks if task["game_id"] not in seen]

    if logging_level == "standard":
        from az.backend_logging import log_backend_selection
        log_backend_selection(
            run_id=run_id,
            workload="tournament",
            requested_backend=backends.get("requested_backend", "auto"),
            backends=backends,
            components_used=components_used,
            primary_path=db_path,
        )
    log(f"run {run_id}: {len(labels)} agents, {config['pairs']} pairs, {len(tasks)} games -> {db_path}")

    for task in tasks:
        task["backends"] = backends
        task.update(run_id=run_id, settings={label: {**settings, "backend": backends} for label, settings in entries.items()},
                    time_budget=time_budget, torch_threads=torch_threads,
                    collect_telemetry=logging_level == "standard")

    started = time.perf_counter()

    def store(record: dict) -> None:
        records.append(record)
        persist_game(record)
        result = "draw" if record["winner"] == "draw" else f"{record['winner_agent']} wins"
        if record["illegal_by"]:
            result += f" (illegal move by {record['illegal_by']})"
        log(f"[{len(records)}/{len(tasks)}] {record['first_player']} (black) vs {record['second_player']} (white): "
            f"{result}, {record['moves_count']} moves, {record['duration_sec']:.1f}s")

    if workers == 1:
        for task in tasks:
            store(play_game_task(task, builder))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(play_game_task, task) for task in tasks]
            for future in as_completed(futures):
                store(future.result())

    records.sort(key=lambda r: r["game_id"])
    result = aggregate(records, labels, time_budget)
    hashes_after = {path: _sha256(path) for path in checkpoints}
    changed = sorted(path for path in checkpoints if hashes_before[path] != hashes_after[path])
    illegal_total = sum(1 for r in records if r["illegal_by"])
    warnings = []
    if changed:
        warnings.append(f"checkpoint changed during the run: {changed}")
    if illegal_total:
        warnings.append(f"{illegal_total} game(s) ended after an illegal move; inspect the SQLite game payloads")
    for row in result["pairs"]:
        if row["games_a_black"] != row["games_a_white"]:
            warnings.append(f"colours not split evenly for {row['agent_a']} vs {row['agent_b']}")
        if games >= 4 and max(row["distinct_games_a_black"], row["distinct_games_b_black"]) == 1:
            warnings.append(f"{row['agent_a']} vs {row['agent_b']}: every game with the same colours was identical "
                            "(both agents deterministic); consider --random-opening")

    summary = {
        **config,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_sec": round(time.perf_counter() - started, 3),
        "games_played": len(records),
        "illegal_moves_total": illegal_total,
        "checkpoint_unchanged": not changed,
        "warnings": warnings,
        "matrix": result["matrix"],
        "agents_summary": result["agents"],
        "pairs_summary": result["pairs"],
    }
    if logging_level == "standard":
        with connect_tournament_db(db_path) as connection:
            existing = connection.execute(
                "SELECT payload_json FROM events WHERE experiment_id=? AND event='tournament_summary'",
                (run_id,),
            ).fetchone()
            if existing is not None:
                saved_summary = json.loads(existing["payload_json"])
                keys = ("games_played", "matrix", "agents_summary", "pairs_summary", "warnings")
                if any(saved_summary.get(key) != summary.get(key) for key in keys):
                    raise ValueError(f"completed tournament summary conflicts with SQLite experiment {run_id}")
            else:
                for warning in warnings:
                    write_event(
                        connection, experiment_id=run_id, recorded_at=summary["finished_at"],
                        level="warning", event="tournament_warning", payload={"message": warning},
                    )
                write_event(
                    connection, experiment_id=run_id, recorded_at=summary["finished_at"],
                    event="tournament_summary", payload=summary,
                )
    for warning in warnings:
        log(f"WARNING: {warning}")
    log(f"done in {summary['wall_time_sec']:.1f}s -> {db_path}")
    return db_path, summary


ALIASES = {"az": "az_v1_threat", "mcts_pure": "mcts-pure", "az_renju": "az_v0"}


def resolve_agent(agent: str) -> str:
    if agent == "az_v1":
        raise ValueError("ambiguous az_v1; use az_v1_tactical or az_v1_threat")
    return ALIASES.get(agent, agent)


def parse_budget(value: str) -> tuple[str, int | float]:
    budget = Budget.parse(value)
    return budget.mode, budget.value


def checkpoint_manifest(checkpoints: dict[str, str]) -> dict:
    result = {}
    for name, path in checkpoints.items():
        digest = hashlib.sha256()
        from az.model import resolve_checkpoint_path
        path = str(resolve_checkpoint_path(path))
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}
    return result




def checkpoint_mapping(entries: list[str], agents: list[str], rule_set: str) -> dict[str, str]:
    mapping = {}
    for entry in entries:
        for item in entry.split(","):
            name, separator, path = item.partition("=")
            if not separator or not name or not path:
                raise ValueError("checkpoint mapping must be NAME=PATH")
            if name in mapping:
                raise ValueError(f"duplicate checkpoint mapping: {name}")
            mapping[name] = path
    if any(not resolve_agent(name).startswith("az") for name in mapping):
        raise ValueError("checkpoint mappings require AZ agents")
    unknown = set(mapping) - set(agents) - {resolve_agent(a) for a in agents}
    if unknown:
        raise ValueError(f"checkpoint mapping names not in agents: {sorted(unknown)}")
    result = {}
    for agent in agents:
        if resolve_agent(agent).startswith("az"):
            default = "checkpoints/renju_latest.json" if rule_set == "renju" or agent == "az_renju" else "checkpoints/normal/latest.json"
            result[agent] = mapping.get(agent, mapping.get(resolve_agent(agent), default))
    return result


def make_named_player(agent: str, rules, *, config: str, checkpoint: str | None,
                      seed: int, budget: tuple[str, int | float], depth: int,
                      backends: dict | None = None):
    kind, amount = budget
    mcts_backend = backends.get("mcts", "auto") if backends else "auto"
    patterns_backend = backends.get("patterns", "auto") if backends else "auto"
    player = create_player(
        resolve_agent(agent), rules, config=config, checkpoint=checkpoint, seed=seed,
        rollouts=int(amount) if kind == "sims" else 50, depth=depth,
        n_simulations=int(amount) if kind == "sims" else None,
        time_budget_sec=float(amount) if kind == "time" else None,
        mcts_backend=mcts_backend, patterns_backend=patterns_backend,
    )
    player.name = agent
    return player




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--budget", default="sims:200")
    parser.add_argument("--depth", "--alphabeta-depth", type=int, default=1)
    parser.add_argument("--rules", "--rule-set", "--ruleset", choices=("normal", "renju"), default="normal")
    parser.add_argument("--checkpoint", "--checkpoints", action="append", default=[])
    parser.add_argument("--config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", "--torch-threads", type=int, default=1)
    parser.add_argument("--run-id")
    parser.add_argument("--db", type=Path, default=TOURNAMENT_DB, help="Tournament SQLite database.")
    parser.add_argument("--logging-level", choices=("none", "standard"), default="standard",
                        help="NONE disables telemetry and durable logs for baselines; STANDARD persists tournament data.")
    parser.add_argument("--pair")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--random-opening", type=int, default=0)
    parser.add_argument("--alphabeta-no-pruning", action="store_true")
    parser.add_argument("--mcts-rollouts-time-mode", type=int, default=200)
    parser.add_argument("--time-sims-cap", type=int, default=100000)
    parser.add_argument("--seed-scheme", choices=("paired-crc32", "legacy-v1"), default="paired-crc32")
    parser.add_argument("--backend", choices=("auto", "cpp", "python"), default="auto")
    parser.add_argument("--mcts-backend", choices=("python", "cpp"))
    parser.add_argument("--patterns-backend", choices=("python", "cpp"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.agents:
            raise ValueError("--agents is required")
        specs = parse_agents(args.agents)
        labels = [spec.label for spec in specs]
        if "az_renju" in labels and args.rules != "renju":
            raise ValueError("az_renju requires --rules renju")
        budget = Budget.parse(args.budget)
        if args.games < 2 or args.games % 2 or args.depth <= 0 or args.threads < 0 or args.workers < 1:
            raise ValueError("games must be positive and even; depth/workers positive; threads nonnegative")
        if args.random_opening < 0 or args.mcts_rollouts_time_mode <= 0 or args.time_sims_cap <= 0:
            raise ValueError("invalid opening or search cap")
        config_path = args.config or f"configs/{args.rules}.yaml"
        loaded_config = load_config(config_path)
        shared = [entry for entry in args.checkpoint if "=" not in entry]
        if len(shared) > 1:
            raise ValueError("only one shared checkpoint may be supplied")
        mappings = checkpoint_mapping([entry for entry in args.checkpoint if "=" in entry], labels, args.rules)
        entries = {}
        for spec in specs:
            default_checkpoint = shared[0] if shared else mappings.get(spec.label)
            defaults = Defaults(checkpoint=default_checkpoint, config=config_path, alphabeta_depth=args.depth,
                                alphabeta_pruning=not args.alphabeta_no_pruning,
                                mcts_rollouts_time_mode=args.mcts_rollouts_time_mode, time_sims_cap=args.time_sims_cap)
            entries[spec.label] = agent_settings(spec, budget, defaults)
        kinds = {settings["factory"]["agent"] for settings in entries.values()}
        mcts_backend = args.mcts_backend
        patterns_backend = args.patterns_backend
        if args.backend == "auto":
            if mcts_backend is None and not kinds & set(AZ_KINDS):
                mcts_backend = "python"
            if patterns_backend is None and not kinds & {"alphabeta", "heuristic", "az_v1_threat"}:
                patterns_backend = "python"
        from az.backend import configure_backends
        backends = configure_backends(
            args.backend,
            mcts_backend=mcts_backend,
            patterns_backend=patterns_backend,
            allow_build=not args.dry_run,
        )
        if args.logging_level == "standard":
            print(json.dumps({"backends": backends}, sort_keys=True))
        if args.dry_run:
            for label, settings in entries.items():
                print(label, settings["budget_note"])
            pairs = len(specs) * (len(specs) - 1) // 2
            print(f"{pairs} pairs x {args.games} games = {pairs * args.games} games (dry run, nothing played)")
            return 0
        _, summary = run_tournament(entries, games=args.games, budget=budget, ruleset=args.rules, seed=args.seed,
                                    run_id=args.run_id,
                                    workers=args.workers, opening_plies=args.random_opening,
                                    torch_threads=args.threads or None, resume=args.resume, pair=args.pair,
                                    backends=backends, db_path=args.db, logging_level=args.logging_level)
        return 0 if summary["illegal_moves_total"] == 0 and summary["checkpoint_unchanged"] else 2
    except (ImportError, RuntimeError, ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
