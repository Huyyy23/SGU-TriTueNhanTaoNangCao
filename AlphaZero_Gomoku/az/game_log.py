from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


VALID_END_REASONS = {"five_in_row", "board_full", "forbidden_move", "timeout"}
VALID_WINNERS = {"black", "white", "draw"}


@dataclass
class MoveRecord:
    """Record for an individual move (active only when log_moves is enabled)."""
    move_index: int
    player: int             # 1 for Black, 2 for White
    row: int
    col: int
    value_before: float
    policy_top_k: list[dict[str, Any]]  # [{"move": idx, "prob": p}, ...]
    simulations: int


@dataclass
class GameRecord:
    """Standardized record for a single game, stored as one JSONL line."""
    game_id: int
    timestamp: str
    rule_set: str
    black_player: str
    white_player: str
    winner: str             # "black" | "white" | "draw"
    first_player_won: bool
    end_reason: str         # "five_in_row" | "board_full" | "forbidden_move" | "timeout"
    num_moves: int
    duration_sec: float
    nn_calls: int = 0
    mean_batch: float = 1.0
    model_iter: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    moves: list[MoveRecord] | None = None

    def validate(self) -> None:
        if self.end_reason not in VALID_END_REASONS:
            raise ValueError(f"Invalid end_reason '{self.end_reason}'. Expected one of {VALID_END_REASONS}")
        if self.winner not in VALID_WINNERS:
            raise ValueError(f"Invalid winner '{self.winner}'. Expected one of {VALID_WINNERS}")


class GameLogger:
    """Unified append-only JSONL logger with immediate flushing for games and evaluation."""

    def __init__(self, log_path: str | Path, log_moves: bool = False) -> None:
        self.log_path = Path(log_path)
        self.log_moves = log_moves
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_game(self, record: GameRecord) -> None:
        """Validate and append a game record as a single JSONL line, flushing to disk immediately."""
        record.validate()
        data = asdict(record)
        if not self.log_moves:
            data.pop("moves", None)

        line = json.dumps(data, ensure_ascii=False)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

    @classmethod
    def read_records(cls, log_path: str | Path) -> list[dict[str, Any]]:
        """Read all game records from a JSONL log file."""
        path = Path(log_path)
        if not path.is_file():
            return []
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @classmethod
    def generate_summary(
        cls,
        records: list[dict[str, Any]],
        agent_a: str | None = None,
        agent_b: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate records into a structured summary for reporting."""
        total = len(records)
        if total == 0:
            return {"total_games": 0}

        if agent_a is None:
            agent_a = records[0]["black_player"]
        if agent_b is None:
            agent_b = records[0]["white_player"]

        wins_a = 0
        draws = 0
        black_games = 0
        black_wins_a = 0
        white_games = 0
        white_wins_a = 0

        end_reasons: dict[str, int] = {k: 0 for k in VALID_END_REASONS}
        total_moves = 0
        total_duration = 0.0

        for r in records:
            winner = r["winner"]
            b_player = r["black_player"]
            w_player = r["white_player"]
            end_reasons[r["end_reason"]] = end_reasons.get(r["end_reason"], 0) + 1
            total_moves += r["num_moves"]
            total_duration += r["duration_sec"]

            is_a_black = (b_player == agent_a)
            if is_a_black:
                black_games += 1
                if winner == "black":
                    wins_a += 1
                    black_wins_a += 1
                elif winner == "draw":
                    draws += 1
            else:
                white_games += 1
                if winner == "white":
                    wins_a += 1
                    white_wins_a += 1
                elif winner == "draw":
                    draws += 1

        summary = {
            "total_games": total,
            "agent_a": agent_a,
            "agent_b": agent_b,
            "overall_win_rate_a": round(wins_a / total, 4),
            "draw_rate": round(draws / total, 4),
            "as_black": {
                "games": black_games,
                "wins": black_wins_a,
                "win_rate": round(black_wins_a / black_games, 4) if black_games > 0 else 0.0,
            },
            "as_white": {
                "games": white_games,
                "wins": white_wins_a,
                "win_rate": round(white_wins_a / white_games, 4) if white_games > 0 else 0.0,
            },
            "by_end_reason": end_reasons,
            "mean_game_duration_sec": round(total_duration / total, 4),
            "mean_moves_per_game": round(total_moves / total, 2),
        }
        return summary

    def save_summary(self, summary_path: str | Path, agent_a: str | None = None, agent_b: str | None = None) -> dict[str, Any]:
        """Compute summary from current log file and write summary.json."""
        records = self.read_records(self.log_path)
        summary = self.generate_summary(records, agent_a=agent_a, agent_b=agent_b)
        out_path = Path(summary_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary
