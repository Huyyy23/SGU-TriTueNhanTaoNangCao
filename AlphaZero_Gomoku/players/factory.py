from __future__ import annotations

import random
from pathlib import Path

from gomoku.rules import Rules
from players.base import Player

AGENT_CHOICES = (
    "random",
    "mcts-pure",
    "heuristic",
    "alphabeta",
    "az",
    "az_v0",
    "az_v1_tactical",
    "az_v1_threat",
)

_AZ_AGENTS = {"az", "az_v0", "az_v1_tactical", "az_v1_threat"}


def create_player(
    agent: str,
    rules: Rules,
    *,
    checkpoint: str | Path | None = None,
    config: str | Path | None = None,
    seed: int | None = None,
    rollouts: int = 50,
    depth: int = 2,
    pruning: bool = True,
    time_budget_sec: float | None = None,
) -> Player:
    """Create one configured player for the terminal or Pygame interface."""
    if agent == "random":
        from players.random import RandomPlayer

        return RandomPlayer(rules, seed=seed)
    if agent == "mcts-pure":
        from players.mcts_pure import MCTSPure

        return MCTSPure(rules, n_rollouts=rollouts, rng=random.Random(seed))
    if agent == "heuristic":
        from players.heuristic import HeuristicPlayer

        return HeuristicPlayer(rules)
    if agent == "alphabeta":
        from players.alphabeta import AlphaBetaPlayer

        return AlphaBetaPlayer(rules, depth=depth, pruning=pruning)
    if agent not in _AZ_AGENTS:
        raise ValueError(f"unknown agent {agent!r}; choose from {AGENT_CHOICES}")
    if checkpoint is None:
        raise ValueError(f"agent {agent!r} requires --checkpoint")

    from az.player_factory import wrap_az_player
    from config import load_config
    from players.az_player import AZPlayer

    if config is None:
        config = Path(__file__).resolve().parents[1] / "configs" / f"{rules.name}.yaml"
    cfg = load_config(config)
    cfg["tactical"] = {
        **(cfg.get("tactical") or {}),
        "enabled": agent in {"az_v1_tactical", "az_v1_threat"},
    }
    cfg["threat_space"] = {
        **(cfg.get("threat_space") or {}),
        "enabled": agent == "az_v1_threat",
    }
    base = AZPlayer(
        checkpoint,
        rules,
        cfg=cfg,
        time_budget_sec=time_budget_sec,
        seed=seed,
    )
    return wrap_az_player(base, cfg)
