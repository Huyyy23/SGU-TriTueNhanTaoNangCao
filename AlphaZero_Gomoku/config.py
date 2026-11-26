from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_CONFIG: dict = {
    # Budget
    "n_iter": 20,
    "games_per_iter": 50,
    # Search
    "n_simulations": 200,
    "c_puct": 1.5,
    "dirichlet_alpha": 0.15,
    "dirichlet_eps": 0.25,
    "temp_moves": 15,
    # Training
    "buffer_size": 50000,
    "train_steps_per_iter": 200,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "optimizer": "adam",
    "weight_decay": 1e-4,
    "value_loss_weight": 1.0,
    "policy_loss_weight": 1.0,
    "lr_schedule": "cosine",
    # Model
    "n_blocks": 6,
    "n_filters": 64,
    # Rule set and self-play
    "rule_set": "normal",
    "selfplay_opening_plies": 0,
    # Fraction of self-play games where one side plays uniformly random legal moves.
    # The search still runs and its visit targets are still stored, so the sample
    # contract is unchanged; only the played move and the resulting trajectory differ.
    "selfplay_random_opponent_rate": 0.0,
    # Data-health stop rule: once a run has this many games, stop when one side wins
    # at least this share of them. Disabled while the rate is above 1.0.
    "outcome_guard_min_games": 0,
    "outcome_guard_max_side_win_rate": 1.01,
    "parallel_games": 16,
    "pi_target_temp": 1.0,
    "move_time_budget_sec": None,
    "eval_fixed_batch_size": 512,
    "checkpoint_buffer": True,
    "log_moves": False,
    # Feature flags for the az_v1 milestones (T29/T30). Both stay off by default so that
    # az_v0 is reproduced exactly; players/factory.py turns them on per agent name.
    "tactical": {"enabled": False},
    "threat_space": {"enabled": False},
}


def _validate_selfplay_opening_plies(config: dict) -> None:
    value = config.get("selfplay_opening_plies", 0)
    if type(value) is not int or value not in (0, 2, 4):
        raise ValueError("selfplay_opening_plies must be 0, 2 or 4")


def _validate_selfplay_random_opponent_rate(config: dict) -> None:
    value = config.get("selfplay_random_opponent_rate", 0.0)
    if type(value) not in (int, float) or not 0.0 <= float(value) <= 1.0:
        raise ValueError("selfplay_random_opponent_rate must be a number in [0, 1]")


def _validate_outcome_guard(config: dict) -> None:
    min_games = config.get("outcome_guard_min_games", 0)
    if type(min_games) is not int or min_games < 0:
        raise ValueError("outcome_guard_min_games must be a non-negative integer")
    rate = config.get("outcome_guard_max_side_win_rate", 1.01)
    if type(rate) not in (int, float) or not 0.5 < float(rate) <= 1.01:
        raise ValueError("outcome_guard_max_side_win_rate must be in (0.5, 1.01]")


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config over DEFAULT_CONFIG. A `base:` key inherits one level from a sibling file."""
    # deepcopy: nested sections (tactical, threat_space) must not be shared between callers.
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = data.pop("base", None)
    if base:
        # shortcut: one level of inheritance, enough for normal/renju
        config.update(yaml.safe_load((path.parent / base).read_text(encoding="utf-8")) or {})
    config.update(data)
    _validate_selfplay_opening_plies(config)
    _validate_selfplay_random_opponent_rate(config)
    _validate_outcome_guard(config)
    return config
