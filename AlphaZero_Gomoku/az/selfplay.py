from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import datetime, timezone

import numpy as np

from az.game_log import GameRecord
from az.mcts import MCTS
from az.replay_buffer import ReplayBuffer
from gomoku.board import BLACK, SIZE, Board
from gomoku.features import board_to_planes


def run_games(
    model,
    rules,
    cfg: dict,
    n_games: int,
    seed: int = 0,
    buffer: ReplayBuffer | None = None,
) -> Iterator[GameRecord]:
    """Play `n_games` self-play games one at a time (deterministic mode: batch = 1).

    `model` is any evaluator with `predict(features)` or `predict_batch(requests)`.
    Samples are appended to `buffer` when the game ends, once z is known.
    """
    if cfg["rule_set"] != rules.name:
        raise ValueError(f"config rule_set '{cfg['rule_set']}' does not match rules '{rules.name}'.")

    rng = np.random.default_rng(seed)
    for game_id in range(n_games):
        yield _play_one_game(model, rules, cfg, game_id, rng, buffer)


def _play_one_game(
    model,
    rules,
    cfg: dict,
    game_id: int,
    rng: np.random.Generator,
    buffer: ReplayBuffer | None,
) -> GameRecord:
    board = Board()
    mcts = MCTS(
        rules=rules,
        n_simulations=cfg["n_simulations"],
        c_puct=cfg["c_puct"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
        dirichlet_eps=cfg["dirichlet_eps"],
        root_noise=True,
        rng=rng,
    )

    positions: list[tuple[np.ndarray, np.ndarray, int]] = []
    started = time.perf_counter()
    while not rules.is_terminal(board):
        counts = mcts.search(board, evaluator=model)
        total = int(counts.sum())
        if total == 0:
            raise RuntimeError("MCTS finished without visiting any move.")
        pi = counts.astype(np.float32) ** (1.0 / float(cfg["pi_target_temp"]))
        pi /= pi.sum()
        positions.append((board_to_planes(board, rules, board.current_player), pi, board.current_player))

        temperature = 1.0 if len(positions) <= cfg["temp_moves"] else 0.0
        board.play(*divmod(mcts.select_move(temperature), SIZE))
    duration_sec = time.perf_counter() - started

    won = rules.is_win(board)
    winner = int(board.grid[board.history[-1]]) if won else None

    if buffer is not None:
        for planes, pi, player in positions:
            if winner is None:
                z = 0.0
            else:
                z = 1.0 if player == winner else -1.0
            buffer.add(planes, pi, z, rules.name)

    return GameRecord(
        game_id=game_id,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rule_set=rules.name,
        black_player="selfplay",
        white_player="selfplay",
        winner="draw" if winner is None else ("black" if winner == BLACK else "white"),
        first_player_won=winner == BLACK,
        end_reason="five_in_row" if won else "board_full",
        num_moves=len(positions),
        duration_sec=duration_sec,
        nn_calls=mcts.nn_calls,
        mean_batch=1.0,
    )
