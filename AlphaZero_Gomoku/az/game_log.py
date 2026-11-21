from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_END_REASONS = {"five_in_row", "board_full", "forbidden_move", "timeout"}
VALID_WINNERS = {"black", "white", "draw", "none"}
# "none" is an abandoned or unfinished game; "draw" is a game that ended on a full board.


@dataclass
class MoveRecord:
    """Record for one move and its search telemetry."""
    move_index: int
    player: int             # 1 for Black, 2 for White
    row: int
    col: int
    value_before: float
    policy_top_k: list[dict[str, Any]]  # [{"move": idx, "prob": p}, ...]
    simulations: int
    latency_sec: float = 0.0  # Wall time from root reset through search completion, including batch wait.
    nodes_visited: int | None = None
    simulations_run: int | None = None
    nn_calls: int | None = None  # Evaluated neural leaves, not batched forward-pass calls.
    override_applied: bool = False
    override_reason: str | None = None
    temperature: float | None = None


@dataclass
class GameRecord:
    """Structured record for one completed or abandoned game."""
    game_id: int
    timestamp: str
    rule_set: str
    first_player: str       # Black, who always moves first in Gomoku
    second_player: str      # White
    winner: str             # "black" | "white" | "draw" | "none"
    first_player_won: bool
    end_reason: str         # "five_in_row" | "board_full" | "forbidden_move" | "timeout"
    moves_count: int
    duration_sec: float
    nn_calls: int = 0
    mean_batch: float = 1.0
    model_iter: int = 0
    config: dict[str, Any] = field(default_factory=dict)
    moves: list[MoveRecord] | None = None
    override_count: int = 0
    override_rate: float = 0.0
    root_data: list[dict[str, Any]] | None = None

    def validate(self) -> None:
        if self.end_reason not in VALID_END_REASONS:
            raise ValueError(f"Invalid end_reason '{self.end_reason}'. Expected one of {VALID_END_REASONS}")
        if self.winner not in VALID_WINNERS:
            raise ValueError(f"Invalid winner '{self.winner}'. Expected one of {VALID_WINNERS}")
        if self.moves is not None and self.moves_count != len(self.moves):
            raise ValueError(
                f"moves_count ({self.moves_count}) does not match len(moves) ({len(self.moves)})"
            )


