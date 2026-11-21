"""Query completed tournament evidence from the shared SQLite database."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sqlite3
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from az.logging_v2 import TOURNAMENT_DB  # noqa: E402


def _budget_mode(config: dict) -> str:
    budget = config.get("budget")
    if isinstance(budget, dict):
        return str(budget.get("mode", "unknown"))
    if isinstance(budget, str) and ":" in budget:
        return budget.split(":", 1)[0]
    return "unknown"


def _wilson(successes: float, games: int) -> list[float]:
    proportion = successes / games
    denominator = 1 + 1.96**2 / games
    center = (proportion + 1.96**2 / (2 * games)) / denominator
    half = 1.96 * (proportion * (1 - proportion) / games + 1.96**2 / (4 * games**2)) ** 0.5 / denominator
    return [center - half, center + half]


def analyze(db_path: str | Path = TOURNAMENT_DB, *, experiment_ids=None, seed: int | None = 20261113) -> dict:
    path = Path(db_path).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        if experiment_ids:
            placeholders = ",".join("?" for _ in experiment_ids)
            experiments = connection.execute(
                f"SELECT experiment_id, rule_set, seed, config_json FROM experiments "
                f"WHERE experiment_id IN ({placeholders}) ORDER BY experiment_id",
                tuple(experiment_ids),
            ).fetchall()
            if len(experiments) != len(set(experiment_ids)):
                found = {row["experiment_id"] for row in experiments}
                raise ValueError(f"unknown experiment IDs: {sorted(set(experiment_ids) - found)}")
        else:
            experiments = connection.execute(
                "SELECT experiment_id, rule_set, seed, config_json FROM experiments "
                "WHERE seed IS ? ORDER BY experiment_id",
                (seed,),
            ).fetchall()

        grouped = defaultdict(lambda: defaultdict(list))
        source_hashes = defaultdict(set)
        for experiment in experiments:
            experiment_id = experiment["experiment_id"]
            config = json.loads(experiment["config_json"])
            game_rows = connection.execute(
                "SELECT payload_json FROM games WHERE experiment_id=?",
                (experiment_id,),
            ).fetchall()
            if not game_rows:
                continue
            mode = _budget_mode(config)
            group = f"{experiment['rule_set']}_{'time' if mode == 'time' else 'fixed' if mode in {'sims', 'simulation'} else 'unknown'}"
            history = connection.execute(
                "SELECT payload_json FROM events WHERE experiment_id=? AND event LIKE 'historical_%'",
                (experiment_id,),
            ).fetchall()
            for row in history:
                payload = json.loads(row["payload_json"])
                if payload.get("source_hash"):
                    source_hashes[experiment_id].add(payload["source_hash"])
            for row in game_rows:
                game = json.loads(row["payload_json"])
                pair = game.get("pair")
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(f"game has no two-agent identity: {experiment_id}")
                if game.get("winner") not in {"black", "white", "draw"}:
                    raise ValueError(f"game has unknown winner: {experiment_id}")
                if not isinstance(game.get("moves_count"), int) or not isinstance(game.get("duration_sec"), (int, float)):
                    raise ValueError(f"game is missing result metrics: {experiment_id}")
                grouped[group][tuple(pair)].append((experiment_id, game, config))

    summaries, rankings = {}, {}
    for group, pairs in grouped.items():
        summarized_pairs = []
        agent_totals = defaultdict(Counter)
        for pair, games in pairs.items():
            a, b = pair
            games.sort(key=lambda item: (
                0, int(item[1]["game_id"])
            ) if str(item[1].get("game_id", "")).isdigit() else (1, str(item[1].get("game_id", ""))))
            wins = draws = losses = 0
            colors = Counter()
            moves_by_agent = defaultdict(list)
            for index, (_experiment_id, game, _config) in enumerate(games):
                black, white = game.get("first_player"), game.get("second_player")
                if black not in pair or white not in pair or black == white:
                    raise ValueError(f"game has invalid player/color identity: {game.get('game_id')}")
                if game["winner"] == "draw":
                    draws += 1
                    colors["draws"] += 1
                else:
                    winner = black if game["winner"] == "black" else white
                    colors["black_wins" if game["winner"] == "black" else "white_wins"] += 1
                    if winner == a:
                        wins += 1
                        colors["a_black_wins" if black == a else "a_white_wins"] += 1
                    else:
                        losses += 1
                for move in game.get("raw_moves") or ():
                    player = move.get("player")
                    agent = black if player in (1, "black") else white if player in (2, "white") else None
                    if agent is not None:
                        moves_by_agent[agent].append(move)
            count = len(games)
            if wins + draws + losses != count:
                raise ValueError(f"game reconciliation failed for pair {pair}")
            score = (wins + draws / 2) / count
            telemetry = {}
            for agent, moves in moves_by_agent.items():
                latencies = [move["latency_sec"] for move in moves if isinstance(move.get("latency_sec"), (int, float))]
                telemetry[agent] = {
                    "moves": len(moves),
                    "mean_ms": statistics.mean(latencies) * 1000 if latencies else None,
                    "p95_ms": float(np.percentile(latencies, 95)) * 1000 if latencies else None,
                    "over_1s": sum(value > 1 for value in latencies),
                    "override_count": sum(move.get("override_applied", False) for move in moves),
                }
                for field in ("nodes_visited", "simulations_run", "nn_calls"):
                    values = [move[field] for move in moves if isinstance(move.get(field), (int, float))]
                    telemetry[agent][field + "_mean"] = statistics.mean(values) if values else None
            summary = {
                "a": a, "b": b, "games": count, "W": wins, "D": draws, "L": losses,
                "score": score, "wilson95": _wilson(wins + draws / 2, count),
                "colors": dict(colors),
                "mean_game_sec": statistics.mean(game["duration_sec"] for _, game, _ in games),
                "mean_moves": statistics.mean(game["moves_count"] for _, game, _ in games),
                "telemetry": telemetry,
                "experiment_ids": sorted({experiment_id for experiment_id, _, _ in games}),
                "source_hashes": sorted({digest for experiment_id, _, _ in games for digest in source_hashes[experiment_id]}),
            }
            summarized_pairs.append(summary)
            for agent, result in ((a, (wins, draws, losses)), (b, (losses, draws, wins))):
                agent_totals[agent].update(W=result[0], D=result[1], L=result[2])
        summaries[group] = summarized_pairs
        rankings[group] = [
            {"agent": agent, **dict(counts),
             "score": (counts["W"] + counts["D"] / 2) / sum(counts.values())}
            for agent, counts in sorted(
                agent_totals.items(),
                key=lambda item: -(item[1]["W"] + item[1]["D"] / 2) / sum(item[1].values()),
            )
        ]
    return {
        "experiments": [row["experiment_id"] for row in experiments],
        "games": sum(pair["games"] for group in summaries.values() for pair in group),
        "groups": summaries,
        "rankings": rankings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=TOURNAMENT_DB)
    parser.add_argument("--experiment-id", action="append")
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--out", type=Path, help="Optional JSON export; no file is written by default.")
    args = parser.parse_args(argv)
    result = analyze(args.db, experiment_ids=args.experiment_id, seed=None if args.experiment_id else args.seed)
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
