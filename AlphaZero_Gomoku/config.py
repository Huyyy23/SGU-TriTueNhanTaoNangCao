from __future__ import annotations

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
    # Evaluation
    "arena_win_threshold": 0.55,
    # Model
    "n_blocks": 6,
    "n_filters": 64,
    # Rule set and self-play
    "rule_set": "normal",
    "parallel_games": 16,
    "pi_target_temp": 1.0,
    "move_time_budget_sec": None,
    "eval_fixed_batch_size": 512,
    "checkpoint_buffer": True,
    "log_moves": False,
}


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config over DEFAULT_CONFIG. A `base:` key inherits one level from a sibling file."""
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = data.pop("base", None)
    if base:
        # shortcut: one level of inheritance, enough for normal/renju
        config.update(yaml.safe_load((path.parent / base).read_text(encoding="utf-8")) or {})
    config.update(data)
    return config
