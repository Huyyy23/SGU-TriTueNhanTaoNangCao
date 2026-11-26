"""Data-health checks for self-play runs.

A self-play run can keep producing games while its outcome distribution collapses to
one side. Training on that data teaches a value head that only knows "the first
player wins", so the run should stop and the data pipeline should be fixed instead of
spending more games. These helpers are pure and do not import torch.
"""

from __future__ import annotations

from collections.abc import Mapping


def outcome_counts(connection, run_id: str) -> dict[str, int]:
    """Winner counts for one run, read from the self-play database."""
    rows = connection.execute(
        "SELECT winner, count(*) FROM games WHERE run_id=? GROUP BY winner", (run_id,)
    ).fetchall()
    return {str(winner): int(count) for winner, count in rows}


def outcome_imbalance_message(
    counts: Mapping[str, int],
    *,
    min_games: int = 0,
    max_side_win_rate: float = 1.01,
) -> str | None:
    """Return a stop message when one side wins at least `max_side_win_rate`.

    `min_games` avoids stopping on the first few games. A rate above 1.0 disables the
    guard. Draws count toward the denominator, so a draw-heavy run never trips it.
    """
    black = int(counts.get("black", 0))
    white = int(counts.get("white", 0))
    total = black + white + int(counts.get("draw", 0))
    if total == 0 or total < int(min_games) or float(max_side_win_rate) > 1.0:
        return None
    side, wins = ("black", black) if black >= white else ("white", white)
    rate = wins / total
    if rate < float(max_side_win_rate):
        return None
    return (
        f"degenerate self-play outcome: {side} won {wins}/{total} games ({rate:.1%}), "
        f"at or above the {float(max_side_win_rate):.0%} guard. Stop and fix data "
        "diversity before spending more games."
    )
