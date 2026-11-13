"""Round-robin tournament between N agents, colours split evenly (T31).

Example:
    python scripts/tournament.py --agents random,heuristic,alphabeta --games 20 --budget sims:200

Every run writes a new folder logs/tournament/<run_id>/ with:
    config.json   run settings, written before the first game
    games.jsonl   one line per finished game, appended as games end
    matrix.csv    score of the row agent against the column agent
    pairs.csv     one row per pair: W/D/L, score, Wilson CI, score by colour, timing
    summary.json  everything above plus per-agent timing, nodes/sims and tactical metrics

Budget (`--budget`):
    sims:N  az_* use N PUCT simulations per move; mcts_pure uses N rollouts per move;
            alphabeta searches a fixed depth (--alphabeta-depth), N does not apply;
            random and heuristic do no search, the budget does not apply.
    time:S  az_* stop searching after S seconds per move (simulations capped by --time-sims-cap);
            alphabeta and mcts_pure have no time control in players/, so they keep their fixed
            depth / rollouts and the summary counts the moves that went over S;
            random and heuristic: not applicable.

Speed: each game process runs torch with one thread (--torch-threads 1). On CPU, one-position
inference with torch's default thread pool was about 10x slower in our smoke test; use
--workers to play several games at once instead.

The script only reads checkpoints (their SHA-256 is checked before and after) and never
reuses an existing run folder. Every game is rebuilt from its own seed, so a run can be
reproduced game by game with the same command. There is no resume: an interrupted run
keeps its games.jsonl, but must be started again under a new run id.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import platform
import random
import subprocess
import sys
import time
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from az.evaluator import wilson_interval
from gomoku.board import BLACK, EMPTY, SIZE
from gomoku.rules import get_rules
from gomoku.runner import Runner
from players.base import IllegalMoveError

AZ_KINDS = ("az", "az_v0", "az_v1_tactical", "az_v1_threat")
KNOWN_KINDS = ("random", "heuristic", "alphabeta", "mcts-pure") + AZ_KINDS
# Names used in tasks/todo.md map onto the factory names in players/factory.py.
ALIASES = {"mcts_pure": "mcts-pure", "az_v1": "az_v1_tactical"}
SPEC_KEYS = {"sims", "depth", "pruning", "checkpoint"}


# -- budget ------------------------------------------------------------------


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
        if value <= 0:
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
        return {"factory": factory, "budget_applies": True, "budget_enforced": False,
                "budget_note": (f"MCTSPure has no time control: fixed {factory['rollouts']} rollouts per move; "
                                f"moves over {budget.value:g} s are counted")}
    if kind == "alphabeta":
        factory["depth"] = int(params.get("depth", defaults.alphabeta_depth))
        factory["pruning"] = bool(params.get("pruning", defaults.alphabeta_pruning))
        if budget.mode == "sims":
            note = f"fixed depth {factory['depth']} (pruning {'on' if factory['pruning'] else 'off'}); sims:N does not apply"
        else:
            note = (f"AlphaBetaPlayer has no time control: fixed depth {factory['depth']}; "
                    f"moves over {budget.value:g} s are counted")
        return {"factory": factory, "budget_applies": budget.mode == "sims", "budget_enforced": False,
                "budget_note": note}
    # random, heuristic: no search at all.
    return {"factory": factory, "budget_applies": False, "budget_enforced": False,
            "budget_note": "no search; the budget does not apply"}


def default_builder(settings: dict, rules, seed: int):
    """Build a player through the shared factory used by scripts/play.py and the UI."""
    from players.factory import create_player

    kwargs = dict(settings["factory"])
    agent = kwargs.pop("agent")
    return create_player(agent, rules, seed=seed, **kwargs)


# -- instrumentation -----------------------------------------------------------


class Probe:
    """Wrap a player: time every move and read its search counters without changing it."""

    def __init__(self, player, label: str, opening: list[int] | None = None) -> None:
        self.player = player
        self.name = label
        self.rules = getattr(player, "rules", None)
        self.opening = opening or []
        self.moves: list[dict] = []
        self._nn_calls = 0
        self._base_time = 0.0
        self._threat: dict | None = None

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
        self._nn_calls, self._base_time, self._threat = 0, 0.0, None
        started = time.perf_counter()
        move = self.player.select_move(board)
        elapsed = time.perf_counter() - started
        record: dict[str, Any] = {"t": elapsed}
        if hasattr(self.player, "nodes_visited"):
            record["nodes"] = int(self.player.nodes_visited)
        if hasattr(self.player, "n_rollouts"):
            record["sims"] = int(self.player.n_rollouts)
        if self._nn_calls:
            record["nn_calls"] = self._nn_calls
        if hasattr(self.player, "last_override_applied"):
            record["override"] = self.player.last_override_reason if self.player.last_override_applied else None
            record["extra_t"] = max(0.0, elapsed - self._base_time)
        if self._threat is not None:
            record["threat"] = self._threat
        self.moves.append(record)
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
    _set_torch_threads(task.get("torch_threads"))
    builder = builder or default_builder
    rules = get_rules(task["ruleset"])
    opening = task["opening"]
    black_label, white_label = task["black"], task["white"]
    black = Probe(builder(task["settings"][black_label], rules, task["seeds"]["black"]), black_label, opening)
    white = Probe(builder(task["settings"][white_label], rules, task["seeds"]["white"]), white_label, opening)

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
        "opening_plies": len(opening),
        "duration_sec": round(time.perf_counter() - started, 4),
        "seeds": task["seeds"],
        "stats": {
            "black": _move_stats(black.moves, budget_time),
            "white": _move_stats(white.moves, budget_time),
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


# -- output files --------------------------------------------------------------


def write_matrix_csv(path: Path, labels: list[str], result: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["agent"] + labels + ["overall_score"])
        for row in labels:
            cells = ["" if col == row else _fmt(result["matrix"][row][col]) for col in labels]
            writer.writerow([row] + cells + [_fmt(result["agents"][row]["score"])])


PAIR_COLUMNS = [
    "agent_a", "agent_b", "games", "wins_a", "draws", "wins_b", "score_a", "wilson95_low_a", "wilson95_high_a",
    "score_a_black", "score_a_white", "score_b_black", "score_b_white", "games_a_black", "games_a_white",
    "distinct_games_a_black", "distinct_games_b_black", "illegal_moves_a", "illegal_moves_b", "mean_moves_per_game",
    "mean_time_a_sec", "p95_time_a_sec", "mean_time_b_sec", "p95_time_b_sec",
    "mean_nodes_a", "mean_nodes_b", "nn_calls_per_move_a", "nn_calls_per_move_b",
]


def write_pairs_csv(path: Path, result: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in result["pairs"]:
            sa, sb = row["stats_a"], row["stats_b"]
            ci = row["wilson95_a"] or [None, None]
            writer.writerow({
                **{k: _fmt(row[k]) for k in PAIR_COLUMNS if k in row},
                "wilson95_low_a": _fmt(ci[0]), "wilson95_high_a": _fmt(ci[1]),
                "mean_time_a_sec": _fmt(sa["mean_time_per_move_sec"]), "p95_time_a_sec": _fmt(sa["p95_time_per_move_sec"]),
                "mean_time_b_sec": _fmt(sb["mean_time_per_move_sec"]), "p95_time_b_sec": _fmt(sb["p95_time_per_move_sec"]),
                "mean_nodes_a": _fmt(sa.get("mean_nodes_per_move")), "mean_nodes_b": _fmt(sb.get("mean_nodes_per_move")),
                "nn_calls_per_move_a": _fmt(sa.get("mean_nn_calls_per_move")),
                "nn_calls_per_move_b": _fmt(sb.get("mean_nn_calls_per_move")),
            })


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}" if abs(value) >= 1e-3 or value == 0 else f"{value:.6f}"
    return str(value)


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
    out_root: str | Path = "logs/tournament",
    run_id: str | None = None,
    workers: int = 1,
    opening_plies: int = 0,
    builder: Callable | None = None,
    torch_threads: int | None = 1,
    extra_config: dict | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[Path, dict]:
    """Play the round-robin for `entries` (label -> resolved settings) and write all files."""
    log = log or (lambda message: print(message, flush=True))
    labels = list(entries)
    get_rules(ruleset)  # fail fast on an unknown rule set
    if workers < 1:
        raise ValueError("--workers must be >= 1")
    if workers > 1 and builder is not None:
        raise ValueError("a custom builder only works with workers=1")
    if opening_plies < 0:
        raise ValueError("--random-opening must be >= 0")
    run_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}_{ruleset}_{budget.label}_s{seed}"
    run_dir = Path(out_root) / run_id
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} already exists; logs are never overwritten, pick another --run-id")

    tasks = schedule(labels, games, seed, ruleset, opening_plies)
    time_budget = budget.value if budget.mode == "time" else None
    checkpoints = sorted({s["factory"]["checkpoint"] for s in entries.values() if s["factory"].get("checkpoint")})
    hashes_before = {path: _sha256(path) for path in checkpoints}

    # Build every agent once before the first game: a bad checkpoint fails here, not mid-run.
    _set_torch_threads(torch_threads)
    rules = get_rules(ruleset)
    for label, settings in entries.items():
        (builder or default_builder)(settings, rules, seed)

    run_dir.mkdir(parents=True)
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
        "seed_rule": "crc32(seed|agent_a|agent_b|pair_game//2|A or B or opening); colour-swapped games share seeds",
        "random_opening_plies": opening_plies,
        "score_rule": "win = 1, draw = 0.5, loss = 0; Wilson 95% interval on (wins + 0.5 * draws) / games",
        "illegal_move_rule": "an illegal move loses the game and is counted in illegal_moves_*",
        "workers": workers,
        "torch_threads_per_game": torch_threads,
        "checkpoint_sha256": hashes_before,
        "git": _git_state(),
        "versions": _versions(),
        **(extra_config or {}),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"run {run_id}: {len(labels)} agents, {config['pairs']} pairs, {len(tasks)} games -> {run_dir}")

    for task in tasks:
        task.update(run_id=run_id, settings=entries, time_budget=time_budget, torch_threads=torch_threads)

    started = time.perf_counter()
    records: list[dict] = []
    games_path = run_dir / "games.jsonl"

    def store(record: dict) -> None:
        records.append(record)
        with open(games_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
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
        warnings.append(f"{illegal_total} game(s) ended on an illegal move; see games.jsonl")
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
    write_matrix_csv(run_dir / "matrix.csv", labels, result)
    write_pairs_csv(run_dir / "pairs.csv", result)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for warning in warnings:
        log(f"WARNING: {warning}")
    log(f"done in {summary['wall_time_sec']:.1f}s -> {run_dir}")
    return run_dir, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Round-robin tournament with colours split evenly (T31).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("agents: random, heuristic, alphabeta, mcts_pure, az_v0, az_v1_tactical, az_v1_threat\n"
                "per-agent options: alphabeta@depth=2, alphabeta@pruning=false, mcts_pure@sims=50,\n"
                "                   az_v0@checkpoint=checkpoints/renju_last.pt (join several with @)"),
    )
    parser.add_argument("--agents", required=True, help="Comma-separated list, e.g. random,heuristic,alphabeta")
    parser.add_argument("--games", type=int, default=100, help="Games per pair, even (half with each colour).")
    parser.add_argument("--budget", default="sims:200", help="sims:N or time:S (see module docstring).")
    parser.add_argument("--ruleset", choices=("normal", "renju"), default="normal")
    parser.add_argument("--checkpoint", help="Checkpoint shared by all az_* agents (e.g. checkpoints/last.pt).")
    parser.add_argument("--config", help="AlphaZero YAML; defaults to configs/<ruleset>.yaml.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alphabeta-depth", type=int, default=1,
                        help="Fixed Alpha-Beta depth (default 1: depth 2 takes minutes per move on CPU).")
    parser.add_argument("--alphabeta-no-pruning", action="store_true")
    parser.add_argument("--mcts-rollouts-time-mode", type=int, default=200,
                        help="Rollouts for mcts_pure under --budget time:S (it has no time control).")
    parser.add_argument("--time-sims-cap", type=int, default=100_000,
                        help="Upper bound on az_* simulations under --budget time:S.")
    parser.add_argument("--random-opening", type=int, default=0, metavar="PLIES",
                        help="Start each colour-swapped pair of games from the same random opening (default 0 = off).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel game processes (default 1).")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="torch threads per game process (default 1: single-position inference is "
                             "several times slower with more threads on CPU; 0 = torch default).")
    parser.add_argument("--out-root", default="logs/tournament")
    parser.add_argument("--run-id", help="Folder name under --out-root; must not exist yet.")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved agents and game count, play nothing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        specs = parse_agents(args.agents)
        budget = Budget.parse(args.budget)
        if args.games < 2 or args.games % 2:
            raise ValueError("--games must be an even number >= 2 so colours split evenly")
        if args.torch_threads < 0:
            raise ValueError("--torch-threads must be >= 0")
        if args.alphabeta_depth <= 0 or args.mcts_rollouts_time_mode <= 0 or args.time_sims_cap <= 0:
            raise ValueError("--alphabeta-depth, --mcts-rollouts-time-mode and --time-sims-cap must be positive")
        if args.config and not Path(args.config).is_file():
            raise ValueError(f"--config {args.config!r} not found")
        defaults = Defaults(
            checkpoint=args.checkpoint,
            config=args.config,
            alphabeta_depth=args.alphabeta_depth,
            alphabeta_pruning=not args.alphabeta_no_pruning,
            mcts_rollouts_time_mode=args.mcts_rollouts_time_mode,
            time_sims_cap=args.time_sims_cap,
        )
        entries = {spec.label: agent_settings(spec, budget, defaults) for spec in specs}
    except ValueError as exc:
        parser.error(str(exc))

    n_pairs = len(specs) * (len(specs) - 1) // 2
    for label, settings in entries.items():
        print(f"  {label:<28} {settings['budget_note']}")
    if args.dry_run:
        print(f"{n_pairs} pairs x {args.games} games = {n_pairs * args.games} games (dry run, nothing played)")
        return 0
    try:
        _, summary = run_tournament(
            entries,
            games=args.games,
            budget=budget,
            ruleset=args.ruleset,
            seed=args.seed,
            out_root=args.out_root,
            run_id=args.run_id,
            workers=args.workers,
            opening_plies=args.random_opening,
            torch_threads=args.torch_threads or None,
        )
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    return 0 if summary["illegal_moves_total"] == 0 and summary["checkpoint_unchanged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
