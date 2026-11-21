from __future__ import annotations

import hashlib
import json
import uuid
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import torch

from az.game_log import GameRecord, MoveRecord
from az.mcts import LeafRequest, MCTS, NativeMCTS
from az.backend import configure_backends
from az.logging_v2 import encode_mcts_root
from az.replay_buffer import ReplayBuffer
from config import _validate_selfplay_opening_plies
from gomoku.board import BLACK, SIZE, Board
from gomoku.features import board_to_planes


@dataclass
class _Game:
    """One interleaved self-play game: its own board, tree, and pending samples."""

    game_id: int
    provenance: dict
    board: Board
    mcts: MCTS
    started: float
    positions: list[tuple[np.ndarray, np.ndarray, int]] = field(default_factory=list)
    mean_batch: float = 1.0
    mcts_logging: str = "NONE"
    search_started: float = 0.0
    move_records: list[MoveRecord] | None = None
    root_data: list[dict] | None = None

def _tensor_digest(state_dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _game_provenance(cfg, seed, model_iter, game_id, worker_digest):
    base = dict(cfg.get("provenance") or {})
    run_id = base.get("run_id") or cfg["_selfplay_run_id"]
    identity = json.dumps([run_id, model_iter, seed, game_id], separators=(",", ":")).encode()
    game_hash = hashlib.sha256(identity).hexdigest()
    seed_identity = json.dumps([seed, model_iter], separators=(",", ":")).encode()
    game_seed = (int.from_bytes(hashlib.sha256(seed_identity).digest()[:8], "big") + game_id) % (1 << 64)
    return {**base, "run_id": run_id, "iteration": model_iter, "game_id": game_hash,
            "game_seed": game_seed, "worker_model_sha256": worker_digest,
            "source_checkpoint_sha256": base.get("source_checkpoint_sha256"), "ruleset": cfg["rule_set"]}


def _model_digest(model):
    return _tensor_digest(model.state_dict()) if hasattr(model, "state_dict") else None


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
    _validate_selfplay_opening_plies(cfg)
    if cfg["rule_set"] != rules.name:
        raise ValueError(f"config rule_set '{cfg['rule_set']}' does not match rules '{rules.name}'.")
    backends = cfg.get("backend_metadata")
    if backends is None:
        backends = configure_backends(
            cfg.get("backend", "auto"),
            mcts_backend=cfg.get("mcts_backend"),
            patterns_backend=cfg.get("patterns_backend", "python"),
            allow_build=cfg.get("_allow_native_build"),
        )
    cfg["mcts_backend"] = backends["mcts"]
    mcts_logging = str(cfg.get("mcts_logging_level", "NONE")).upper()
    if mcts_logging not in {"NONE", "STANDARD"}:
        raise ValueError(f"unsupported MCTS logging level: {mcts_logging}")
    cfg["mcts_logging_level"] = mcts_logging
    cfg["patterns_backend"] = backends["patterns"]
    cfg["backend_metadata"] = backends
    cfg = {**cfg, "_selfplay_run_id": cfg.get("_selfplay_run_id", uuid.uuid4().hex)}
    worker_digest = _model_digest(model)
    expected_digest = cfg.get("_expected_model_sha256")
    if expected_digest is not None and worker_digest != expected_digest:
        raise ValueError("Worker model tensor digest does not match parent snapshot.")
    parallel = max(1, int(cfg["parallel_games"]))
    next_game_id = 0
    active: list[_Game] = []
    batch_calls = 0
    batch_leaves = 0

    while next_game_id < n_games or active:
        while len(active) < parallel and next_game_id < n_games:
            game_id = next_game_id + int(cfg.get("_game_offset", 0))
            provenance = _game_provenance(cfg, seed, model_iter, game_id, worker_digest)
            active.append(_new_game(game_id, rules, cfg, np.random.default_rng(provenance["game_seed"]), provenance))
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


def _new_game(game_id: int, rules, cfg: dict, rng: np.random.Generator, provenance: dict) -> _Game:
    board = Board()
    opening_actions = []
    for _ in range(cfg.get("selfplay_opening_plies", 0)):
        action = int(rng.choice(rules.legal_moves(board)))
        opening_actions.append(action)
        board.play(*divmod(action, SIZE))
    provenance["opening_actions"] = opening_actions
    mcts_cls = NativeMCTS if cfg.get("mcts_backend", "python") == "cpp" else MCTS
    mcts = mcts_cls(
        rules=rules,
        n_simulations=cfg["n_simulations"],
        c_puct=cfg["c_puct"],
        dirichlet_alpha=cfg["dirichlet_alpha"],
        dirichlet_eps=cfg["dirichlet_eps"],
        root_noise=True,
        rng=rng,
    )
    mcts.reset(board)
    logging_enabled = cfg.get("mcts_logging_level", "NONE") == "STANDARD"
    return _Game(
        game_id=game_id, provenance=provenance, board=board, mcts=mcts, started=time.perf_counter(),
        mcts_logging="STANDARD" if logging_enabled else "NONE",
        search_started=time.perf_counter() if logging_enabled else 0.0,
        move_records=[] if logging_enabled else None,
        root_data=[] if logging_enabled else None,
    )


def _advance(
    game: _Game,
    rules,
    cfg: dict,
    buffer: ReplayBuffer | None,
    model_iter: int,
    samples_out: list | None = None,
) -> GameRecord | None:
    """Pick a move for one finished search. Return a record when the game is over."""
    logging_enabled = game.mcts_logging == "STANDARD"
    search_finished = time.perf_counter() if logging_enabled else 0.0
    if logging_enabled:
        counts, priors, q_values, legal_mask = game.mcts.root_statistics()
    else:
        counts = game.mcts.visit_counts()
    if int(counts.sum()) == 0:
        raise RuntimeError("MCTS finished without visiting any move.")

    pi = counts.astype(np.float32) ** (1.0 / float(cfg["pi_target_temp"]))
    pi /= pi.sum()
    player = game.board.current_player
    move_index = len(game.board.history)
    game.positions.append((board_to_planes(game.board, rules, player), pi, player))

    temperature = 1.0 if len(game.board.history) < cfg["temp_moves"] else 0.0
    root_value = game.mcts.root_value() if logging_enabled else 0.0
    selected_action = game.mcts.select_move(temperature)
    if logging_enabled:
        forbidden_mask = np.zeros(SIZE * SIZE, dtype=bool)
        forbidden = rules.forbidden_moves(game.board) if hasattr(rules, "forbidden_moves") else ()
        forbidden_mask[forbidden] = True
        root_policy = counts.astype(np.float64)
        root_policy /= root_policy.sum()
        encoding, data = encode_mcts_root(
            counts, root_policy, pi, priors, q_values, legal_mask, forbidden_mask
        )
        game.root_data.append({"move_index": move_index, "encoding": encoding, "data": data})
        row, col = divmod(selected_action, SIZE)
        game.move_records.append(MoveRecord(
            move_index=move_index, player=player, row=row, col=col, value_before=root_value,
            policy_top_k=[], simulations=cfg["n_simulations"], temperature=temperature,
            latency_sec=search_finished - game.search_started,
            simulations_run=game.mcts.simulations_done,
        ))
    game.board.play(*divmod(selected_action, SIZE))

    if not rules.is_terminal(game.board):
        game.mcts.reset(game.board)
        if logging_enabled:
            game.search_started = time.perf_counter()
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
            buffer.add(planes, pi, z, rules.name, game_id=game.provenance["game_id"])
        if samples_out is not None:
            sample = (planes, pi, z)
            samples_out.append((*sample, game.provenance["game_id"]) if cfg.get("_lineage_samples") else sample)

    return GameRecord(
        game_id=game.game_id,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rule_set=rules.name,
        first_player="selfplay",
        second_player="selfplay",
        winner="draw" if winner is None else ("black" if winner == BLACK else "white"),
        first_player_won=winner == BLACK,
        end_reason="five_in_row" if won else "board_full",
        moves_count=len(game.positions),  # MCTS moves; full history also includes opening_actions.
        duration_sec=time.perf_counter() - game.started,
        nn_calls=game.mcts.nn_calls,
        mean_batch=game.mean_batch,
        model_iter=model_iter,
        config={**{k: v for k, v in cfg.items() if not k.startswith("_")}, "provenance": game.provenance,
                "components_used": cfg.get("components_used", {"mcts": cfg.get("mcts_backend", "python"), "patterns": "not_used"})},
        moves=game.move_records,
        root_data=game.root_data,
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
    _validate_selfplay_opening_plies(cfg)

    backends = cfg.get("backend_metadata")
    if backends is None:
        backends = configure_backends(
            cfg.get("backend", "auto"),
            mcts_backend=cfg.get("mcts_backend"),
            patterns_backend=cfg.get("patterns_backend", "python"),
        )
    worker_cfg = {
        **cfg,
        "mcts_backend": backends["mcts"],
        "patterns_backend": backends["patterns"],
        "backend_metadata": backends,
        "_allow_native_build": False,
    }
    state_dict = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    expected_digest = _tensor_digest(state_dict)
    run_id = (cfg.get("provenance") or {}).get("run_id") or uuid.uuid4().hex
    payloads = []
    offset = 0
    for chunk in split_games(n_games, workers):
        payloads.append((state_dict, rules.name, {**worker_cfg,
                         "_game_offset": offset, "_selfplay_run_id": run_id,
                         "_expected_model_sha256": expected_digest, "_lineage_samples": True},
                         chunk, seed, model_iter))
        offset += chunk

    records: list[GameRecord] = []
    samples: list[tuple] = []
    for chunk_records, chunk_samples in pool.map(_play_chunk, payloads):
        records.extend(chunk_records)
        samples.extend(chunk_samples)


    if buffer is not None:
        for planes, pi, z, game_id in samples:
            buffer.add(planes, pi, z, rules.name, game_id=game_id)
    return records


def _play_chunk(payload: tuple) -> tuple[list[GameRecord], list[tuple]]:
    """Worker entry point. Must stay module-level so spawn can pickle it."""
    state_dict, rule_set, cfg, n_games, seed, model_iter = payload

    import torch

    torch.set_num_threads(1)

    if cfg.get("patterns_backend"):
        from gomoku.patterns import set_backend
        set_backend(cfg["patterns_backend"])
    from az.model import AZNet
    from gomoku.rules import get_rules

    rules = get_rules(rule_set)

    model = AZNet(n_blocks=cfg["n_blocks"], n_filters=cfg["n_filters"])
    model.load_state_dict(state_dict)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    samples: list[tuple] = []
    records = list(run_games(model, rules, cfg, n_games, seed=seed, samples_out=samples, model_iter=model_iter))
    return records, samples
