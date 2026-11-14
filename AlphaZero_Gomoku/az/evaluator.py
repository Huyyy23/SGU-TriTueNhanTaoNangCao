from __future__ import annotations

import math
import time
from numbers import Integral
from dataclasses import asdict
from datetime import datetime, timezone

from az.game_log import GameLogger, GameRecord, MoveRecord
from gomoku.board import BLACK, SIZE, Board


def wilson_interval(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval; fractional successes encode draws as half a point."""
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def play_game(black, white, rules, game_id: int, rule_set: str | None = None) -> GameRecord:
    """Play one game to the end and return its log record."""
    board = Board()
    started = time.perf_counter()
    moves = []
    while not rules.is_terminal(board):
        player = black if board.current_player == BLACK else white
        color = board.current_player
        move_started = time.perf_counter()
        row, col = player.select_move(board)
        latency = time.perf_counter() - move_started
        if (not isinstance(row, Integral) or not isinstance(col, Integral)
                or not 0 <= row < SIZE or not 0 <= col < SIZE
                or row * SIZE + col not in rules.legal_moves(board)):
            raise ValueError(f"{getattr(player, 'name', 'player')} returned illegal move {(row, col)} under {rules.name}")
        telemetry = player
        simulations = getattr(telemetry, "simulations_run", None)
        moves.append(MoveRecord(
            move_index=len(moves), player=color, row=int(row), col=int(col),
            value_before=0.0, policy_top_k=[], simulations=simulations or 0,
            latency_sec=latency, nodes_visited=getattr(telemetry, "nodes_visited", None),
            simulations_run=simulations,
            nn_calls=getattr(telemetry, "nn_calls", None),
            override_applied=bool(getattr(player, "last_override_applied", False)),
            override_reason=getattr(player, "last_override_reason", None),
        ))
        board.play(row, col)

    if rules.is_win(board):
        last_mover = int(board.grid[board.history[-1]])
        winner = "black" if last_mover == BLACK else "white"
        end_reason = "five_in_row"
    else:
        winner, end_reason = "draw", "board_full"

    return GameRecord(
        game_id=game_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        rule_set=rule_set or rules.name,
        first_player=getattr(black, "name", "player"),
        second_player=getattr(white, "name", "player"),
        winner=winner,
        first_player_won=winner == "black",
        end_reason=end_reason,
        moves_count=len(board.history),
        duration_sec=round(time.perf_counter() - started, 4),
        nn_calls=sum(m.nn_calls or 0 for m in moves),
        moves=moves,
        override_count=sum(m.override_applied for m in moves),
        override_rate=sum(m.override_applied for m in moves) / len(moves) if moves else 0.0,
    )


def evaluate(a, b, rules, n_games: int, logger: GameLogger | None = None) -> list[GameRecord]:
    """Play `n_games` between a and b, alternating colours: even ids have a as Black."""
    records = []
    for game_id in range(n_games):
        black, white = (a, b) if game_id % 2 == 0 else (b, a)
        record = play_game(black, white, rules, game_id)
        records.append(record)
        if logger is not None:
            logger.log_game(record)
    return records


def summarise(records: list[GameRecord], agent_a: str, agent_b: str) -> dict:
    """Aggregate WDL, score, colors, raw-move latency and actual search work."""
    summary = GameLogger.generate_summary([asdict(r) for r in records], agent_a=agent_a, agent_b=agent_b)
    wins = sum(r.winner == ("black" if r.first_player == agent_a else "white") for r in records)
    draws = sum(r.winner == "draw" for r in records)
    losses = sum(r.winner in {"black", "white"} for r in records) - wins
    n = len(records)
    summary.update(wins=wins, draws=draws, losses=losses, score=(wins + draws / 2) / n if n else 0.0,
                   wilson_95_a=wilson_interval(wins, n), score_ci_95=wilson_interval(wins + draws / 2, n))
    for key, is_black in (("as_black", True), ("as_white", False)):
        subset = [r for r in records if (r.first_player == agent_a) == is_black]
        color_wins = sum(r.winner == ("black" if is_black else "white") for r in subset)
        color_draws = sum(r.winner == "draw" for r in subset)
        count = len(subset)
        summary[key] = dict(games=count, wins=color_wins, draws=color_draws,
                            losses=count - color_wins - color_draws,
                            win_rate=color_wins / count if count else 0.0,
                            score=(color_wins + color_draws / 2) / count if count else 0.0)
    summary["telemetry"] = {}
    for agent in (agent_a, agent_b):
        moves = [m for r in records for m in (r.moves or [])
                 if (r.first_player if m.player == BLACK else r.second_player) == agent]
        latencies = sorted(m.latency_sec for m in moves)
        nodes = [m.nodes_visited for m in moves if m.nodes_visited is not None]
        sims = [m.simulations_run for m in moves if m.simulations_run is not None]
        calls = [m.nn_calls for m in moves if m.nn_calls is not None]
        overrides = sum(m.override_applied for m in moves)
        summary["telemetry"][agent] = dict(
            moves=len(moves), mean_latency_sec=sum(latencies) / len(latencies) if latencies else 0.0,
            p95_latency_sec=latencies[max(0, math.ceil(.95 * len(latencies)) - 1)] if latencies else 0.0,
            nodes_visited=sum(nodes) if nodes else None, simulations_run=sum(sims) if sims else None,
            nn_calls=sum(calls) if calls else None, override_count=overrides,
            override_rate=overrides / len(moves) if moves else 0.0,
        )
    return summary
