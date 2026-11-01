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
    samples_out: list | None = None,
) -> Iterator[GameRecord]:
    """Play `n_games` self-play games, interleaving `cfg["parallel_games"]` of them.

    Each tick advances every unfinished game by exactly one simulation, then evaluates
    all pending leaves with a single `model.predict` call. Games are sampled into
    `buffer` when they end, once z is known; `samples_out` collects the same
    `(planes, pi, z)` tuples for a caller that stores them elsewhere (see
    `run_games_parallel`). `model` needs `predict(features)`.

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
            record = _advance(game, rules, cfg, buffer, model_iter, samples_out)
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
    samples_out: list | None = None,
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

    return _finish_game(game, rules, cfg, buffer, model_iter, samples_out)


def _finish_game(
    game: _Game,
    rules,
    cfg: dict,
    buffer: ReplayBuffer | None,
    model_iter: int,
    samples_out: list | None = None,
) -> GameRecord:
    won = rules.is_win(game.board)
    winner = int(game.board.grid[game.board.history[-1]]) if won else None

    for planes, pi, player in game.positions:
        z = 0.0 if winner is None else (1.0 if player == winner else -1.0)
        if buffer is not None:
            buffer.add(planes, pi, z, rules.name)
        if samples_out is not None:
            samples_out.append((planes, pi, z))

    return GameRecord(
        game_id=game.game_id,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rule_set=rules.name,
        first_player="selfplay",
        second_player="selfplay",
        winner="draw" if winner is None else ("black" if winner == BLACK else "white"),
        first_player_won=winner == BLACK,
        end_reason="five_in_row" if won else "board_full",
        moves_count=len(game.positions),
        duration_sec=time.perf_counter() - game.started,
        nn_calls=game.mcts.nn_calls,
        mean_batch=game.mean_batch,
        model_iter=model_iter,
        config=dict(cfg),
    )


def split_games(n_games: int, workers: int) -> list[int]:
    """Split `n_games` into `workers` near-equal chunks, biggest first. Never more chunks than games."""
    if n_games <= 0:
        return []
    workers = max(1, min(int(workers), int(n_games)))
    base, extra = divmod(int(n_games), workers)
    return [base + 1] * extra + [base] * (workers - extra)


def run_games_parallel(
    pool,
    model,
    rules,
    cfg: dict,
    n_games: int,
    seed: int = 0,
    buffer: ReplayBuffer | None = None,
    model_iter: int = 0,
    workers: int = 1,
) -> list[GameRecord]:
    """Play `n_games` in `workers` processes, then store every sample in `buffer` here.

    Only the parent writes to `buffer`, so a sample can neither be stored twice nor lost.
    `pool` is a `ProcessPoolExecutor` owned by the caller and reused across iterations;
    weights reach the workers as a pickled CPU `state_dict`, so nothing touches disk.
    """
    if cfg["rule_set"] != rules.name:
        raise ValueError(f"config rule_set '{cfg['rule_set']}' does not match rules '{rules.name}'.")

    state_dict = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    payloads = [
        (state_dict, rules.name, {**cfg, "parallel_games": chunk}, chunk, seed * 1000 + index, model_iter)
        for index, chunk in enumerate(split_games(n_games, workers))
    ]

    records: list[GameRecord] = []
    samples: list[tuple] = []
    for chunk_records, chunk_samples in pool.map(_play_chunk, payloads):
        records.extend(chunk_records)
        samples.extend(chunk_samples)

    # Every worker numbers its games from 0; renumber so log ids stay unique.
    for game_id, record in enumerate(records):
        record.game_id = game_id

    if buffer is not None:
        for planes, pi, z in samples:
            buffer.add(planes, pi, z, rules.name)
    return records


def _play_chunk(payload: tuple) -> tuple[list[GameRecord], list[tuple]]:
    """Worker entry point. Must stay module-level so spawn can pickle it."""
    state_dict, rule_set, cfg, n_games, seed, model_iter = payload

    import torch

    torch.set_num_threads(1)

    from az.model import AZNet
    from gomoku.rules.normal import NormalRules

    if rule_set != "normal":
        raise ValueError(f"rule_set '{rule_set}' has no rules implementation yet (T24).")

    model = AZNet(n_blocks=cfg["n_blocks"], n_filters=cfg["n_filters"])
    model.load_state_dict(state_dict)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    samples: list[tuple] = []
    records = list(run_games(model, NormalRules(), cfg, n_games, seed=seed, samples_out=samples, model_iter=model_iter))
    return records, samples
