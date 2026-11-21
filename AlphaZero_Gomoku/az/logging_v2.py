"""Versioned SQLite persistence primitives for Gomoku research telemetry."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
SELFPLAY_DB = Path("data/selfplay.sqlite")
TOURNAMENT_DB = Path("data/tournament.sqlite")

_SELFPLAY_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    git_commit TEXT,
    engine_version TEXT,
    backend TEXT NOT NULL,
    rule_set TEXT NOT NULL,
    board_size INTEGER NOT NULL,
    seed INTEGER,
    config_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    local_game_id TEXT NOT NULL,
    rule_set TEXT NOT NULL,
    winner TEXT NOT NULL,
    end_reason TEXT NOT NULL,
    moves_count INTEGER NOT NULL,
    duration_sec REAL NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(run_id, local_game_id)
);
CREATE INDEX IF NOT EXISTS idx_selfplay_games_run ON games(run_id);
CREATE TABLE IF NOT EXISTS moves (
    game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    move_index INTEGER NOT NULL,
    player INTEGER NOT NULL,
    action INTEGER NOT NULL,
    latency_sec REAL,
    selected_action INTEGER,
    root_value REAL,
    simulations INTEGER,
    backend TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(game_id, move_index)
);
CREATE TABLE IF NOT EXISTS mcts_roots (
    game_id TEXT NOT NULL,
    move_index INTEGER NOT NULL,
    encoding TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY(game_id, move_index),
    FOREIGN KEY(game_id, move_index) REFERENCES moves(game_id, move_index) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS mcts_traces (
    trace_id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL,
    move_index INTEGER NOT NULL,
    encoding TEXT NOT NULL,
    data BLOB NOT NULL,
    FOREIGN KEY(game_id, move_index) REFERENCES moves(game_id, move_index) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS training_steps (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    iteration INTEGER NOT NULL,
    global_step INTEGER NOT NULL,
    recorded_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    PRIMARY KEY(run_id, global_step)
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(run_id, kind, path, sha256)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    run_id TEXT REFERENCES runs(run_id),
    recorded_at TEXT NOT NULL,
    level TEXT NOT NULL,
    event TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_selfplay_events_run_time ON events(run_id, recorded_at);
CREATE TABLE IF NOT EXISTS source_imports (
    source_hash TEXT NOT NULL,
    source_identity TEXT NOT NULL,
    source_path TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(source_hash, source_identity)
);
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""

_TOURNAMENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    git_commit TEXT,
    engine_version TEXT,
    backend TEXT NOT NULL,
    rule_set TEXT NOT NULL,
    board_size INTEGER NOT NULL,
    seed INTEGER,
    config_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS agents (
    agent_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    checkpoint_sha256 TEXT,
    backend TEXT,
    rule_set TEXT NOT NULL,
    config_json TEXT NOT NULL,
    UNIQUE(label, algorithm, checkpoint_sha256, backend, rule_set, config_json)
);
CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    agent_a_id TEXT NOT NULL REFERENCES agents(agent_id),
    agent_b_id TEXT NOT NULL REFERENCES agents(agent_id),
    pair_id TEXT NOT NULL,
    rule_set TEXT NOT NULL,
    seed INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(experiment_id, pair_id)
);
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    match_id TEXT REFERENCES matches(match_id),
    local_game_id TEXT NOT NULL,
    black_agent_id TEXT NOT NULL REFERENCES agents(agent_id),
    white_agent_id TEXT NOT NULL REFERENCES agents(agent_id),
    rule_set TEXT NOT NULL,
    winner TEXT NOT NULL,
    end_reason TEXT NOT NULL,
    moves_count INTEGER NOT NULL,
    duration_sec REAL NOT NULL,
    payload_json TEXT NOT NULL,
    UNIQUE(experiment_id, local_game_id)
);
CREATE INDEX IF NOT EXISTS idx_tournament_games_experiment ON games(experiment_id);
CREATE INDEX IF NOT EXISTS idx_tournament_games_agents ON games(black_agent_id, white_agent_id);
CREATE TABLE IF NOT EXISTS moves (
    game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    move_index INTEGER NOT NULL,
    player INTEGER NOT NULL,
    action INTEGER NOT NULL,
    latency_sec REAL,
    selected_action INTEGER,
    root_value REAL,
    simulations INTEGER,
    backend TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(game_id, move_index)
);
CREATE TABLE IF NOT EXISTS mcts_roots (
    game_id TEXT NOT NULL,
    move_index INTEGER NOT NULL,
    encoding TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY(game_id, move_index),
    FOREIGN KEY(game_id, move_index) REFERENCES moves(game_id, move_index) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS mcts_traces (
    trace_id INTEGER PRIMARY KEY,
    game_id TEXT NOT NULL,
    move_index INTEGER NOT NULL,
    encoding TEXT NOT NULL,
    data BLOB NOT NULL,
    FOREIGN KEY(game_id, move_index) REFERENCES moves(game_id, move_index) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS rating_systems (
    rating_system_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    rule_set TEXT NOT NULL,
    method TEXT NOT NULL,
    config_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rating_events (
    rating_event_id INTEGER PRIMARY KEY,
    rating_system_id TEXT NOT NULL REFERENCES rating_systems(rating_system_id),
    sequence INTEGER NOT NULL,
    game_id TEXT REFERENCES games(game_id),
    agent_id TEXT NOT NULL REFERENCES agents(agent_id),
    rating_before REAL NOT NULL,
    rating_after REAL NOT NULL,
    result REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    UNIQUE(rating_system_id, sequence, agent_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    experiment_id TEXT REFERENCES experiments(experiment_id),
    recorded_at TEXT NOT NULL,
    level TEXT NOT NULL,
    event TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_tournament_events_experiment_time ON events(experiment_id, recorded_at);
CREATE TABLE IF NOT EXISTS source_imports (
    source_hash TEXT NOT NULL,
    source_identity TEXT NOT NULL,
    source_path TEXT NOT NULL,
    imported_at TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    status TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY(source_hash, source_identity)
);
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a foreign-key-checked WAL connection; callers own and close it."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    return connection


def initialize(path: str | Path, database: str) -> None:
    """Create the initial schema; reject unversioned or unsupported databases untouched."""
    if database not in {"selfplay", "tournament"}:
        raise ValueError("database must be 'selfplay' or 'tournament'")
    db_path = Path(path)
    schema = _SELFPLAY_SCHEMA if database == "selfplay" else _TOURNAMENT_SCHEMA
    version = None
    if db_path.exists() and db_path.stat().st_size > 0:
        with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as reader:
            migration_table = reader.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
            if migration_table:
                version = reader.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
            else:
                tables = reader.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' LIMIT 1"
                ).fetchone()
                if tables:
                    raise RuntimeError("refusing to initialize a non-empty unversioned database")
    if version is not None and version > SCHEMA_VERSION:
        raise RuntimeError(f"database schema {version} is newer than supported {SCHEMA_VERSION}")
    if version is not None and version < SCHEMA_VERSION:
        raise RuntimeError(f"database schema {version} has no available migration to {SCHEMA_VERSION}")
    with connect(db_path) as connection:
        connection.executescript("BEGIN IMMEDIATE;" + chr(10) + schema)
        if version is None:
            from datetime import datetime, timezone
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, datetime.now(timezone.utc).isoformat()),
            )
def record_selfplay_run(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    started_at: str,
    backend: str,
    rule_set: str,
    seed: int | None,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO runs(run_id, started_at, backend, rule_set, board_size, seed, config_json, metadata_json) "
        "VALUES (?, ?, ?, ?, 11, ?, ?, ?)",
        (run_id, started_at, backend, rule_set, seed, json_dumps(config), json_dumps(metadata or {})),
    )
    row = connection.execute(
        "SELECT started_at, backend, rule_set, board_size, seed, config_json, metadata_json FROM runs WHERE run_id=?",
        (run_id,),
    ).fetchone()
    expected = (started_at, backend, rule_set, 11, seed, json_dumps(config), json_dumps(metadata or {}))
    if row is None or tuple(row) != expected:
        raise ValueError(f"run_id already exists with different metadata: {run_id}")


def record_selfplay_game(
    connection: sqlite3.Connection, run_id: str, record: dict[str, Any]
) -> str:
    local_id = str(record["game_id"])
    game_id = ((record.get("config") or {}).get("provenance") or {}).get("game_id") or f"{run_id}:{local_id}"
    payload = json_dumps({key: value for key, value in record.items() if key != "root_data"})
    connection.execute(
        "INSERT OR IGNORE INTO games(game_id, run_id, local_game_id, rule_set, winner, end_reason, "
        "moves_count, duration_sec, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (game_id, run_id, local_id, record["rule_set"], record["winner"], record["end_reason"],
         record["moves_count"], record["duration_sec"], payload),
    )
    existing = connection.execute("SELECT payload_json FROM games WHERE game_id=?", (game_id,)).fetchone()
    if existing is None or existing[0] != payload:
        raise ValueError(f"game identity collision with different data: {game_id}")
    for move in record.get("moves") or ():
        move_data = dict(move)
        action = int(move_data["row"]) * 11 + int(move_data["col"])
        connection.execute(
            "INSERT OR IGNORE INTO moves(game_id, move_index, player, action, latency_sec, selected_action, "
            "root_value, simulations, backend, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (game_id, move_data["move_index"], move_data["player"], action,
             move_data.get("latency_sec"), action, move_data.get("value_before"),
             move_data.get("simulations_run", move_data.get("simulations")),
             (record.get("config") or {}).get("mcts_backend"), json_dumps(move_data)),
        )
        saved = connection.execute(
            "SELECT payload_json FROM moves WHERE game_id=? AND move_index=?",
            (game_id, move_data["move_index"]),
        ).fetchone()
        if saved is None or saved[0] != json_dumps(move_data):
            raise ValueError(f"move identity collision: {game_id}/{move_data['move_index']}")
    for root in record.get("root_data") or ():
        move_index = int(root["move_index"])
        encoding, data = root["encoding"], root["data"]
        connection.execute(
            "INSERT OR IGNORE INTO mcts_roots(game_id, move_index, encoding, data) VALUES (?, ?, ?, ?)",
            (game_id, move_index, encoding, data),
        )
        saved = connection.execute(
            "SELECT encoding, data FROM mcts_roots WHERE game_id=? AND move_index=?",
            (game_id, move_index),
        ).fetchone()
        if saved is None or tuple(saved) != (encoding, data):
            raise ValueError(f"MCTS root identity collision: {game_id}/{move_index}")
    return game_id


def record_training_step(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    iteration: int,
    global_step: int,
    recorded_at: str,
    metrics: dict[str, Any],
) -> None:
    connection.execute(
        "INSERT INTO training_steps(run_id, iteration, global_step, recorded_at, metrics_json) "
        "VALUES (?, ?, ?, ?, ?) ON CONFLICT(run_id, global_step) DO NOTHING",
        (run_id, iteration, global_step, recorded_at, json_dumps(metrics)),
    )
    saved = connection.execute(
        "SELECT iteration, recorded_at, metrics_json FROM training_steps WHERE run_id=? AND global_step=?",
        (run_id, global_step),
    ).fetchone()
    if saved is None or tuple(saved) != (iteration, recorded_at, json_dumps(metrics)):
        raise ValueError(f"training step identity collision: {run_id}/{global_step}")

def record_tournament_experiment(
    connection: sqlite3.Connection,
    *,
    experiment_id: str,
    started_at: str,
    backend: str,
    rule_set: str,
    seed: int | None,
    config: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO experiments(experiment_id, started_at, backend, rule_set, board_size, seed, config_json, metadata_json) "
        "VALUES (?, ?, ?, ?, 11, ?, ?, ?)",
        (experiment_id, started_at, backend, rule_set, seed, json_dumps(config), json_dumps(metadata or {})),
    )
    row = connection.execute(
        "SELECT started_at, backend, rule_set, board_size, seed, config_json, metadata_json "
        "FROM experiments WHERE experiment_id=?",
        (experiment_id,),
    ).fetchone()
    expected = (started_at, backend, rule_set, 11, seed, json_dumps(config), json_dumps(metadata or {}))
    if row is None or tuple(row) != expected:
        raise ValueError(f"experiment_id already exists with different metadata: {experiment_id}")


def tournament_agent_id(
    *,
    label: str,
    algorithm: str,
    checkpoint_sha256: str | None,
    backend: str | None,
    rule_set: str,
    config: dict[str, Any],
) -> str:
    identity = {
        "label": label,
        "algorithm": algorithm,
        "checkpoint_sha256": checkpoint_sha256,
        "backend": backend,
        "rule_set": rule_set,
        "config_json": json_dumps(config),
    }
    return "agent:" + hashlib.sha256(json_dumps(identity).encode("utf-8")).hexdigest()


def record_tournament_agent(
    connection: sqlite3.Connection,
    *,
    agent_id: str,
    label: str,
    algorithm: str,
    checkpoint_sha256: str | None,
    backend: str | None,
    rule_set: str,
    config: dict[str, Any],
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO agents(agent_id, label, algorithm, checkpoint_sha256, backend, rule_set, config_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (agent_id, label, algorithm, checkpoint_sha256, backend, rule_set, json_dumps(config)),
    )
    row = connection.execute(
        "SELECT label, algorithm, checkpoint_sha256, backend, rule_set, config_json FROM agents WHERE agent_id=?",
        (agent_id,),
    ).fetchone()
    if row is None or tuple(row) != (label, algorithm, checkpoint_sha256, backend, rule_set, json_dumps(config)):
        raise ValueError(f"agent_id already exists with different metadata: {agent_id}")


def record_tournament_match(
    connection: sqlite3.Connection,
    *,
    match_id: str,
    experiment_id: str,
    agent_a_id: str,
    agent_b_id: str,
    pair_id: str,
    rule_set: str,
    seed: int | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    connection.execute(
        "INSERT OR IGNORE INTO matches(match_id, experiment_id, agent_a_id, agent_b_id, pair_id, rule_set, seed, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (match_id, experiment_id, agent_a_id, agent_b_id, pair_id, rule_set, seed, json_dumps(metadata or {})),
    )
    row = connection.execute(
        "SELECT experiment_id, agent_a_id, agent_b_id, pair_id, rule_set, seed, metadata_json "
        "FROM matches WHERE match_id=?",
        (match_id,),
    ).fetchone()
    expected = (
        experiment_id, agent_a_id, agent_b_id, pair_id, rule_set, seed,
        json_dumps(metadata or {}),
    )
    if row is None or tuple(row) != expected:
        raise ValueError(f"match_id already exists with different identity: {match_id}")


def record_tournament_game(
    connection: sqlite3.Connection,
    *,
    experiment_id: str,
    local_game_id: str | int,
    match_id: str,
    black_agent_id: str,
    white_agent_id: str,
    record: dict[str, Any],
) -> str:
    game_id = f"{experiment_id}:{local_game_id}"
    payload = json_dumps(record)
    actions = record.get("moves", ())
    connection.execute(
        "INSERT OR IGNORE INTO games(game_id, experiment_id, match_id, local_game_id, black_agent_id, white_agent_id, "
        "rule_set, winner, end_reason, moves_count, duration_sec, payload_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (game_id, experiment_id, match_id, str(local_game_id), black_agent_id, white_agent_id,
         record["rule_set"], record["winner"], record["end_reason"], record["moves_count"],
         record["duration_sec"], payload),
    )
    existing = connection.execute("SELECT payload_json FROM games WHERE game_id=?", (game_id,)).fetchone()
    if existing is None or existing[0] != payload:
        raise ValueError(f"game identity collision with different data: {game_id}")
    raw_moves = record.get("raw_moves", ())
    for index, action in enumerate(actions):
        raw = raw_moves[index] if index < len(raw_moves) else {}
        move_player = 1 if index % 2 == 0 else 2
        move_payload = json_dumps(raw)
        connection.execute(
            "INSERT OR IGNORE INTO moves(game_id, move_index, player, action, latency_sec, selected_action, "
            "root_value, simulations, backend, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (game_id, index, move_player, int(action), raw.get("latency_sec"), int(action),
             raw.get("value_before"), raw.get("simulations_run"), raw.get("backend"), move_payload),
        )
        saved = connection.execute(
            "SELECT action, payload_json FROM moves WHERE game_id=? AND move_index=?", (game_id, index)
        ).fetchone()
        if saved is None or saved[0] != int(action) or saved[1] != move_payload:
            raise ValueError(f"move identity collision: {game_id}/{index}")
    return game_id


def write_event(
    connection: sqlite3.Connection,
    *,
    event: str,
    payload: dict[str, Any],
    level: str = "info",
    run_id: str | None = None,
    experiment_id: str | None = None,
    recorded_at: str,
) -> None:
    if run_id is not None:
        connection.execute(
            "INSERT INTO events(run_id, recorded_at, level, event, payload_json) VALUES (?, ?, ?, ?, ?)",
            (run_id, recorded_at, level, event, json_dumps(payload)),
        )
    else:
        connection.execute(
            "INSERT INTO events(experiment_id, recorded_at, level, event, payload_json) VALUES (?, ?, ?, ?, ?)",
            (experiment_id, recorded_at, level, event, json_dumps(payload)),
        )


def write_artifact(
    connection: sqlite3.Connection,
    *,
    run_id: str,
    kind: str,
    path: str,
    sha256: str | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    metadata_json = json_dumps(metadata or {})
    connection.execute(
        "INSERT OR IGNORE INTO artifacts(run_id, kind, path, sha256, metadata_json) VALUES (?, ?, ?, ?, ?)",
        (run_id, kind, path, sha256, metadata_json),
    )
    saved = connection.execute(
        "SELECT metadata_json FROM artifacts WHERE run_id=? AND kind=? AND path=? AND sha256 IS ?",
        (run_id, kind, path, sha256),
    ).fetchone()
    if saved is None or saved[0] != metadata_json:
        raise ValueError(f"artifact identity collision: {run_id}/{kind}/{path}/{sha256}")
_ROOT_SIZE = 121
_ROOT_HEADER = struct.Struct("<4sH")
_ROOT_ENCODING = "mcts-root-v1+zlib"


def encode_mcts_root(
    visit_counts: Any,
    root_policy: Any,
    target_policy: Any,
    priors: Any,
    q_values: Any,
    legal_mask: Any,
    forbidden_mask: Any,
) -> tuple[str, bytes]:
    arrays = (
        np.asarray(visit_counts),
        np.asarray(root_policy, dtype="<f8"),
        np.asarray(target_policy, dtype="<f8"),
        np.asarray(priors, dtype="<f8"),
        np.asarray(q_values, dtype="<f8"),
        np.asarray(legal_mask, dtype=np.uint8),
        np.asarray(forbidden_mask, dtype=np.uint8),
    )
    if any(array.shape != (_ROOT_SIZE,) for array in arrays):
        raise ValueError("MCTS root arrays must each have 121 entries")
    if np.any(arrays[0] < 0) or not all(np.isfinite(array).all() for array in arrays[1:5]):
        raise ValueError("MCTS root statistics contain invalid values")
    raw = b"".join((
        _ROOT_HEADER.pack(b"GMR1", 1),
        arrays[0].astype("<u4", copy=False).tobytes(),
        *(array.tobytes() for array in arrays[1:5]),
        arrays[5].tobytes(),
        arrays[6].tobytes(),
    ))
    return _ROOT_ENCODING, zlib.compress(raw)


def decode_mcts_root(encoding: str, data: bytes) -> dict[str, np.ndarray]:
    if encoding != _ROOT_ENCODING:
        raise ValueError(f"unsupported MCTS root encoding: {encoding}")
    raw = zlib.decompress(data)
    if len(raw) != _ROOT_HEADER.size + 4 * _ROOT_SIZE + 4 * 8 * _ROOT_SIZE + 2 * _ROOT_SIZE:
        raise ValueError("invalid MCTS root payload size")
    magic, version = _ROOT_HEADER.unpack_from(raw)
    if magic != b"GMR1" or version != 1:
        raise ValueError("unsupported MCTS root payload version")
    offset = _ROOT_HEADER.size
    result = {}
    for name, dtype in (
        ("visit_counts", "<u4"), ("root_policy", "<f8"), ("target_policy", "<f8"),
        ("priors", "<f8"), ("q_values", "<f8"), ("legal_mask", np.bool_),
        ("forbidden_mask", np.bool_),
    ):
        values = np.frombuffer(raw, dtype=dtype, count=_ROOT_SIZE, offset=offset)
        result[name] = values
        offset += values.nbytes
    return result


def encode_blob(data: bytes, *, compress: bool = True) -> tuple[str, bytes]:
    """Encode opaque diagnostic bytes without precision loss."""
    return ("zlib", zlib.compress(data)) if compress else ("raw", data)


def decode_blob(encoding: str, data: bytes) -> bytes:
    if encoding == "raw":
        return data
    if encoding == "zlib":
        return zlib.decompress(data)
    raise ValueError(f"unsupported BLOB encoding: {encoding}")


def json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def backup_database(source: str | Path, destination: str | Path) -> None:
    """Create a consistent SQLite backup through SQLite's online backup API."""
    source_path, destination_path = Path(source), Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("backup destination must differ from source")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source_path, timeout=30) as source_db:
        with sqlite3.connect(destination_path, timeout=30) as destination_db:
            source_db.backup(destination_db)
