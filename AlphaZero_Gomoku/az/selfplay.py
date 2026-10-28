from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from az.game_log import GameRecord
from az.mcts import LeafRequest, MCTS
from az.replay_buffer import ReplayBuffer
from gomoku.board import BLACK, SIZE, Board
from gomoku.features import board_to_planes


@dataclass
class _Game:
    """One interleaved self-play game: its own board, tree, and pending samples."""

    game_id: int
    board: Board
    mcts: MCTS
    started: float
    positions: list[tuple[np.ndarray, np.ndarray, int]] = field(default_factory=list)
    mean_batch: float = 1.0


def run_games(
    model,
    rules,
    cfg: dict,
    n_games: int,
    seed: int = 0,
    buffer: ReplayBuffer | None = None,
    model_iter: int = 0,
) -> Iterator[GameRecord]:
    """Play `n_games` self-play games, interleaving `cfg["parallel_games"]` of them.

    Each tick advances every unfinished game by exactly one simulation, then evaluates
    all pending leaves with a single `model.predict` call. Games are sampled into
    `buffer` when they end, once z is known. `model` needs `predict(features)`.

    `parallel_games: 1` runs the same code path as a single game, in order.
    """
    if cfg["rule_set"] != rules.name:
        raise ValueError(f"config rule_set '{cfg['rule_set']}' does not match rules '{rules.name}'.")

    rng = np.random.default_rng(seed)
    parallel = max(1, int(cfg["parallel_games"]))
    next_game_id = 0
    active: list[_Game] = []
    batch_calls = 0
    batch_leaves = 0

    while next_game_id < n_games or active:
        while len(active) < parallel and next_game_id < n_games:
            active.append(_new_game(next_game_id, rules, cfg, rng))
            next_game_id += 1

        waiting: list[tuple[_Game, LeafRequest]] = []
        searched: list[_Game] = []
        for game in active:
            request = game.mcts.request()
            if request is None:
                searched.append(game)
            else:
                waiting.append((game, request))

        if waiting:
            planes = np.stack([request.planes for _, request in waiting])
            probs, values = model.predict(planes)
            batch_calls += 1
            batch_leaves += len(waiting)
            mean_batch = batch_leaves / batch_calls
            for game in active:
                game.mean_batch = mean_batch
            for (game, _), policy, value in zip(waiting, probs, values):
                game.mcts.receive(policy, float(value))

        for game in searched:
            record = _advance(game, rules, cfg, buffer, model_iter)
            if record is not None:
                active.remove(game)
                yield record


def _new_game(game_id: int, rules, cfg: dict, rng: np.random.Generator) -> _Game:
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
    mcts.reset(board)
    return _Game(game_id=game_id, board=board, mcts=mcts, started=time.perf_counter())


def _advance(
    game: _Game,
    rules,
    cfg: dict,
    buffer: ReplayBuffer | None,
    model_iter: int,
) -> GameRecord | None:
    """Pick a move for one finished search. Return a record when the game is over."""
    counts = game.mcts.visit_counts()
    if int(counts.sum()) == 0:
        raise RuntimeError("MCTS finished without visiting any move.")

    pi = counts.astype(np.float32) ** (1.0 / float(cfg["pi_target_temp"]))
    pi /= pi.sum()
    game.positions.append((board_to_planes(game.board, rules, game.board.current_player), pi, game.board.current_player))

    temperature = 1.0 if len(game.positions) <= cfg["temp_moves"] else 0.0
    game.board.play(*divmod(game.mcts.select_move(temperature), SIZE))

    if not rules.is_terminal(game.board):
        game.mcts.reset(game.board)
        return None

    return _finish_game(game, rules, cfg, buffer, model_iter)


def _finish_game(
    game: _Game,
    rules,
    cfg: dict,
    buffer: ReplayBuffer | None,
    model_iter: int,
) -> GameRecord:
    won = rules.is_win(game.board)
    winner = int(game.board.grid[game.board.history[-1]]) if won else None

    if buffer is not None:
        for planes, pi, player in game.positions:
            z = 0.0 if winner is None else (1.0 if player == winner else -1.0)
            buffer.add(planes, pi, z, rules.name)

    return GameRecord(
        game_id=game.game_id,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rule_set=rules.name,
        black_player="selfplay",
        white_player="selfplay",
        winner="draw" if winner is None else ("black" if winner == BLACK else "white"),
        first_player_won=winner == BLACK,
        end_reason="five_in_row" if won else "board_full",
        num_moves=len(game.positions),
        duration_sec=time.perf_counter() - game.started,
        nn_calls=game.mcts.nn_calls,
        mean_batch=game.mean_batch,
        model_iter=model_iter,
        config=dict(cfg),
    )
