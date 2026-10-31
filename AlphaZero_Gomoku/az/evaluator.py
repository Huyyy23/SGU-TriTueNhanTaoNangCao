from __future__ import annotations

import math
import time
from dataclasses import asdict
from datetime import datetime, timezone

from az.game_log import GameLogger, GameRecord
from gomoku.board import BLACK, Board


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a win rate, valid for small n."""
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
    while not rules.is_terminal(board):
        player = black if board.current_player == BLACK else white
        row, col = player.select_move(board)
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
        black_player=getattr(black, "name", "player"),
        white_player=getattr(white, "name", "player"),
        winner=winner,
        first_player_won=winner == "black",
        end_reason=end_reason,
        num_moves=len(board.history),
        duration_sec=round(time.perf_counter() - started, 4),
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
    """Aggregate records, adding the Wilson interval for agent_a's overall win rate."""
    summary = GameLogger.generate_summary([asdict(r) for r in records], agent_a=agent_a, agent_b=agent_b)
    wins = sum(1 for r in records if r.winner == ("black" if r.black_player == agent_a else "white"))
    summary["wilson_95_a"] = wilson_interval(wins, len(records))
    return summary
