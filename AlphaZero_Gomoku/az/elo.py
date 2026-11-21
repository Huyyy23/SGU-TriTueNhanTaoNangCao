"""Deterministic Elo recalculation from immutable tournament game rows."""
from __future__ import annotations

import math
import sqlite3
from typing import Any

import numpy as np

from az.logging_v2 import json_dumps


def recalculate_elo(
    connection: sqlite3.Connection,
    experiment_id: str,
    *,
    k_factor: float = 32.0,
    initial_rating: float = 1200.0,
) -> dict[str, Any]:
    """Replay one experiment's games in stable ID order; never overwrite rating history."""
    if not math.isfinite(k_factor) or k_factor <= 0:
        raise ValueError("k_factor must be finite and positive")
    if not math.isfinite(initial_rating):
        raise ValueError("initial_rating must be finite")

    experiment = connection.execute(
        "SELECT rule_set FROM experiments WHERE experiment_id=?", (experiment_id,)
    ).fetchone()
    if experiment is None:
        raise KeyError(f"unknown experiment: {experiment_id}")
    rule_set = experiment["rule_set"]
    games = connection.execute(
        "SELECT game_id, local_game_id, black_agent_id, white_agent_id, rule_set, winner "
        "FROM games WHERE experiment_id=? "
        "ORDER BY CAST(local_game_id AS INTEGER), local_game_id, game_id",
        (experiment_id,),
    ).fetchall()
    if not games:
        raise ValueError(f"experiment has no games: {experiment_id}")
    if any(game["rule_set"] != rule_set for game in games):
        raise ValueError("cannot mix rule sets in one rating system")

    config = {
        "initial_rating": float(initial_rating),
        "k_factor": float(k_factor),
        "ordering": "CAST(local_game_id AS INTEGER), local_game_id, game_id",
        "formula": "standard-elo-expected-score-v1",
    }
    rating_system_id = f"{experiment_id}:elo:v1:k{k_factor:g}:r{initial_rating:g}"
    ratings: dict[str, float] = {}
    events = []
    for sequence, game in enumerate(games, 1):
        black, white = game["black_agent_id"], game["white_agent_id"]
        if black == white:
            raise ValueError(f"self-play game cannot rate one agent against itself: {game['game_id']}")
        winner = game["winner"]
        if winner not in {"black", "white", "draw"}:
            raise ValueError(f"invalid winner in game {game['game_id']}: {winner!r}")
        black_before = ratings.setdefault(black, float(initial_rating))
        white_before = ratings.setdefault(white, float(initial_rating))
        expected_black = 1.0 / (1.0 + 10.0 ** ((white_before - black_before) / 400.0))
        black_result = 0.5 if winner == "draw" else float(winner == "black")
        delta = float(k_factor) * (black_result - expected_black)
        black_after, white_after = black_before + delta, white_before - delta
        events.append((sequence, game["game_id"], black, black_before, black_after,
                       black_result, json_dumps({"expected_score": expected_black, "color": "black"})))
        events.append((sequence, game["game_id"], white, white_before, white_after,
                       1.0 - black_result, json_dumps({"expected_score": 1.0 - expected_black, "color": "white"})))
        ratings[black], ratings[white] = black_after, white_after

    connection.execute("SAVEPOINT recalculate_elo")
    try:
        connection.execute(
            "INSERT OR IGNORE INTO rating_systems(rating_system_id, experiment_id, rule_set, method, config_json) "
            "VALUES (?, ?, ?, 'elo', ?)",
            (rating_system_id, experiment_id, rule_set, json_dumps(config)),
        )
        saved_system = connection.execute(
            "SELECT experiment_id, rule_set, method, config_json FROM rating_systems WHERE rating_system_id=?",
            (rating_system_id,),
        ).fetchone()
        if saved_system is None or tuple(saved_system) != (
            experiment_id, rule_set, "elo", json_dumps(config)
        ):
            raise ValueError(f"rating system identity collision: {rating_system_id}")

        for sequence, game_id, agent_id, before, after, result, metadata_json in events:
            connection.execute(
                "INSERT OR IGNORE INTO rating_events(rating_system_id, sequence, game_id, agent_id, "
                "rating_before, rating_after, result, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (rating_system_id, sequence, game_id, agent_id, before, after, result, metadata_json),
            )
            saved_event = connection.execute(
                "SELECT game_id, rating_before, rating_after, result, metadata_json FROM rating_events "
                "WHERE rating_system_id=? AND sequence=? AND agent_id=?",
                (rating_system_id, sequence, agent_id),
            ).fetchone()
            if saved_event is None or tuple(saved_event) != (
                game_id, before, after, result, metadata_json
            ):
                raise ValueError(f"rating history collision: {rating_system_id}/{sequence}/{agent_id}")
        actual_count = connection.execute(
            "SELECT count(*) FROM rating_events WHERE rating_system_id=?", (rating_system_id,)
        ).fetchone()[0]
        if actual_count != len(events):
            raise ValueError(f"rating history length mismatch: {rating_system_id}")
        connection.execute("RELEASE recalculate_elo")
    except Exception:
        connection.execute("ROLLBACK TO recalculate_elo")
        connection.execute("RELEASE recalculate_elo")
        raise

    return {"rating_system_id": rating_system_id, "rule_set": rule_set, "ratings": ratings}


PROMOTION_EVAL_INTERVAL_GAMES = 5000
PROMOTION_EVAL_GAMES_PER_PAIR = 100
PROMOTION_EVAL_OPENING_PLIES = 4
PROMOTION_EVAL_BOOTSTRAPS = 2000


def paired_davidson_ci(
    connection: sqlite3.Connection,
    experiment_ids: list[str],
    candidate_label: str,
    opponent_label: str,
    *,
    bootstrap_samples: int = PROMOTION_EVAL_BOOTSTRAPS,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate paired Davidson ΔElo and its opening-pair bootstrap interval."""
    if not experiment_ids or bootstrap_samples <= 0:
        raise ValueError("experiment_ids and positive bootstrap_samples are required")

    clusters: dict[tuple[str, str], list[int]] = {}
    candidate_colors: dict[tuple[str, str], list[str]] = {}
    for experiment_id in experiment_ids:
        for row in connection.execute(
            "SELECT m.pair_id, g.winner, ba.label AS black_label, wa.label AS white_label "
            "FROM games g JOIN matches m ON m.match_id=g.match_id "
            "JOIN agents ba ON ba.agent_id=g.black_agent_id "
            "JOIN agents wa ON wa.agent_id=g.white_agent_id "
            "WHERE g.experiment_id=? ORDER BY m.pair_id, CAST(g.local_game_id AS INTEGER)",
            (experiment_id,),
        ):
            black, white = row["black_label"], row["white_label"]
            if {black, white} != {candidate_label, opponent_label}:
                continue
            key = (experiment_id, row["pair_id"])
            if row["winner"] == "draw":
                result = 1
            elif row["winner"] == ("black" if black == candidate_label else "white"):
                result = 0
            elif row["winner"] == ("black" if black == opponent_label else "white"):
                result = 2
            else:
                raise ValueError(f"invalid winner in experiment {experiment_id}")
            clusters.setdefault(key, [0, 0, 0])[result] += 1
            candidate_colors.setdefault(key, []).append("black" if black == candidate_label else "white")

    counts = np.asarray(list(clusters.values()), dtype=np.int64)
    if not len(counts) or np.any(counts.sum(axis=1) != 2):
        raise ValueError("each opening cluster must contain exactly two paired games")
    if any(sorted(colors) != ["black", "white"] for colors in candidate_colors.values()):
        raise ValueError("each opening cluster must swap candidate colors")

    def fit(rows: np.ndarray) -> tuple[float, float]:
        wins, draws, losses = rows.sum(axis=0).astype(float)
        delta = math.log((wins + 0.5) / (losses + 0.5))
        tie_parameter = (draws + 0.5) / math.sqrt((wins + 0.5) * (losses + 0.5))
        return 400.0 * delta / math.log(10.0), tie_parameter

    rating, tie_parameter = fit(counts)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sample = counts[rng.integers(0, len(counts), size=len(counts))]
        bootstrap[index] = fit(sample)[0]
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    wins, draws, losses = counts.sum(axis=0).tolist()
    games = int(counts.sum())
    return {
        "candidate": candidate_label,
        "opponent": opponent_label,
        "games": games,
        "opening_pairs": int(len(counts)),
        "wins": int(wins),
        "draws": int(draws),
        "losses": int(losses),
        "score": (wins + draws / 2) / games,
        "delta_elo": rating,
        "ci95": [float(lower), float(upper)],
        "davidson_tie_parameter": tie_parameter,
        "method": "pairwise-Davidson paired-opening-cluster bootstrap",
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }
