"""Screening gate: score one agent in a tournament database against a threshold.

This is the objective-success screen (AlphaZero vs a baseline), not the
candidate-vs-best promotion gate in SPEC 8.3. It reads a finished tournament
database and never plays or trains games.
"""

from __future__ import annotations

import math


def wilson_interval(score: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion in [0, 1]."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0, 1]")
    denominator = 1.0 + z * z / n
    centre = (score + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def score_summary(outcomes: list[float]) -> dict[str, float | int]:
    """Win/draw/loss summary. A draw counts as half a point, matching the tournament score rule."""
    if not outcomes:
        raise ValueError("no games to score")
    wins = sum(1 for value in outcomes if value == 1.0)
    draws = sum(1 for value in outcomes if value == 0.5)
    losses = sum(1 for value in outcomes if value == 0.0)
    if wins + draws + losses != len(outcomes):
        raise ValueError("outcomes must be 1.0 (win), 0.5 (draw) or 0.0 (loss)")
    score = (wins + 0.5 * draws) / len(outcomes)
    low, high = wilson_interval(score, len(outcomes))
    return {
        "games": len(outcomes),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "wilson_low": low,
        "wilson_high": high,
    }


def gate_passes(
    summary: dict, *, min_games: int, min_score: float, basis: str = "score"
) -> bool:
    """Pass only when the sample is large enough and the chosen statistic clears the target."""
    if basis not in ("score", "wilson_low"):
        raise ValueError("basis must be 'score' or 'wilson_low'")
    return (
        int(summary["games"]) >= int(min_games)
        and float(summary[basis]) >= float(min_score)
    )
