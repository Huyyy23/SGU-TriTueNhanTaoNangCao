"""Explicit, lossless tournament experiment export/import; imports only add rows."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from az import logging_v2

BUNDLE_VERSION = 1
TABLES = (
    "experiments", "agents", "matches", "games", "moves", "mcts_roots", "mcts_traces",
    "rating_systems", "rating_events", "events",
)
ORDER_BY = {
    "experiments": "experiment_id",
    "agents": "agent_id",
    "matches": "match_id",
    "games": "game_id",
    "moves": "game_id, move_index",
    "mcts_roots": "game_id, move_index",
    "mcts_traces": "game_id, move_index, trace_id",
    "rating_systems": "rating_system_id",
    "rating_events": "rating_system_id, sequence, agent_id",
    "events": "recorded_at, event_id",
}


def _q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _query_ids(connection, table: str, column: str, values: set[str], order: str) -> list:
    if not values:
        return []
    rows = []
    ids = sorted(values)
    for start in range(0, len(ids), 900):
        chunk = ids[start:start + 900]
        marks = ",".join("?" for _ in chunk)
        rows.extend(connection.execute(
            f"SELECT * FROM {table} WHERE {column} IN ({marks}) ORDER BY {order}", chunk
        ).fetchall())
    return sorted(rows, key=lambda row: tuple(row[key] for key in order.split(", ")))


def _collect(connection, experiment_id: str) -> dict[str, list]:
    experiments = connection.execute(
        "SELECT * FROM experiments WHERE experiment_id=?", (experiment_id,)
    ).fetchall()
    if len(experiments) != 1:
        raise KeyError(f"unknown experiment: {experiment_id}")
    matches = connection.execute(
        "SELECT * FROM matches WHERE experiment_id=? ORDER BY match_id", (experiment_id,)
    ).fetchall()
    games = connection.execute(
        "SELECT * FROM games WHERE experiment_id=? ORDER BY game_id", (experiment_id,)
    ).fetchall()
    rating_systems = connection.execute(
        "SELECT * FROM rating_systems WHERE experiment_id=? ORDER BY rating_system_id", (experiment_id,)
    ).fetchall()
    game_ids = {row["game_id"] for row in games}
    rating_ids = {row["rating_system_id"] for row in rating_systems}
    rating_events = _query_ids(connection, "rating_events", "rating_system_id", rating_ids, ORDER_BY["rating_events"])
    agent_ids = {row["agent_a_id"] for row in matches} | {row["agent_b_id"] for row in matches}
    agent_ids |= {row["black_agent_id"] for row in games} | {row["white_agent_id"] for row in games}
    agent_ids |= {row["agent_id"] for row in rating_events}
    tables = {
        "experiments": experiments,
        "agents": _query_ids(connection, "agents", "agent_id", agent_ids, ORDER_BY["agents"]),
        "matches": matches,
        "games": games,
        "moves": _query_ids(connection, "moves", "game_id", game_ids, ORDER_BY["moves"]),
        "mcts_roots": _query_ids(connection, "mcts_roots", "game_id", game_ids, ORDER_BY["mcts_roots"]),
        "mcts_traces": _query_ids(connection, "mcts_traces", "game_id", game_ids, ORDER_BY["mcts_traces"]),
        "rating_systems": rating_systems,
        "rating_events": rating_events,
        "events": connection.execute(
            "SELECT * FROM events WHERE experiment_id=? ORDER BY recorded_at, event_id", (experiment_id,)
        ).fetchall(),
    }
    return tables


def _agent_fingerprint(row) -> str:
    fields = ("label", "algorithm", "checkpoint_sha256", "backend", "rule_set", "config_json")
    return hashlib.sha256(logging_v2.json_dumps({key: row[key] for key in fields}).encode("utf-8")).hexdigest()


def _content_hash(tables: dict[str, list]) -> str:
    fingerprints = {row["agent_id"]: _agent_fingerprint(row) for row in tables["agents"]}
    digest = hashlib.sha256()
    for table in TABLES:
        for row in tables[table]:
            values = dict(row)
            if table == "agents":
                values["agent_id"] = fingerprints[values["agent_id"]]
            elif table == "matches":
                values["agent_a_id"] = fingerprints[values["agent_a_id"]]
                values["agent_b_id"] = fingerprints[values["agent_b_id"]]
            elif table == "games":
                values["black_agent_id"] = fingerprints[values["black_agent_id"]]
                values["white_agent_id"] = fingerprints[values["white_agent_id"]]
            elif table == "rating_events":
                values["agent_id"] = fingerprints[values["agent_id"]]
            for surrogate in {"events": ("event_id",), "rating_events": ("rating_event_id",),
                              "mcts_traces": ("trace_id",)}.get(table, ()):
                values.pop(surrogate, None)
            canonical = {key: ({"blob_hex": value.hex()} if isinstance(value, bytes) else value)
                         for key, value in sorted(values.items())}
            record = (table + "\0" + json.dumps(
                canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
            ) + "\n").encode("utf-8")
            digest.update(record)
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_experiment(source_db: str | Path, destination: str | Path, experiment_id: str) -> str:
    """Create a new standalone SQLite bundle; refuse to overwrite any existing path."""
    source_path, destination_path = Path(source_db), Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("bundle destination must differ from the source database")

    source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    target = None
    try:
        if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("source database integrity_check failed")
        if source.execute("PRAGMA foreign_key_check").fetchall():
            raise RuntimeError("source database foreign_key_check failed")
        version = source.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        if version != logging_v2.SCHEMA_VERSION:
            raise RuntimeError(f"unsupported source schema version: {version}")
        tables = _collect(source, experiment_id)
        content_hash = _content_hash(tables)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(destination_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
        logging_v2.initialize(destination_path, "tournament")
        target = logging_v2.connect(destination_path)
        with target:
            for table in TABLES:
                rows = tables[table]
                if not rows:
                    continue
                columns = rows[0].keys()
                names = ",".join(_q(column) for column in columns)
                marks = ",".join("?" for _ in columns)
                target.executemany(
                    f"INSERT INTO {_q(table)} ({names}) VALUES ({marks})",
                    [tuple(row) for row in rows],
                )
            target.execute(
                "CREATE TABLE exchange_manifest(bundle_version INTEGER NOT NULL, experiment_id TEXT NOT NULL, "
                "content_sha256 TEXT NOT NULL, created_at TEXT NOT NULL)"
            )
            target.execute(
                "INSERT INTO exchange_manifest VALUES (?, ?, ?, ?)",
                (BUNDLE_VERSION, experiment_id, content_hash, datetime.now(timezone.utc).isoformat()),
            )
    finally:
        source.close()
        if target is not None:
            target.close()

    checkpoint = sqlite3.connect(destination_path)
    try:
        checkpoint.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        checkpoint.execute("PRAGMA journal_mode=DELETE")
    finally:
        checkpoint.close()
    return _file_hash(destination_path)


def _read_bundle(path: Path):
    bundle = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    bundle.row_factory = sqlite3.Row
    try:
        if bundle.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("bundle integrity_check failed")
        if bundle.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("bundle foreign_key_check failed")
        migrations = bundle.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0]
        if migrations != logging_v2.SCHEMA_VERSION:
            raise ValueError(f"unsupported bundle schema version: {migrations}")
        if bundle.execute("SELECT count(*) FROM exchange_manifest").fetchone()[0] != 1:
            raise ValueError("bundle must contain exactly one exchange manifest")
        manifest = bundle.execute(
            "SELECT bundle_version, experiment_id, content_sha256 FROM exchange_manifest"
        ).fetchone()
        if manifest["bundle_version"] != BUNDLE_VERSION:
            raise ValueError("unsupported exchange manifest version")
        experiment_id = manifest["experiment_id"]
        tables = _collect(bundle, experiment_id)
        if any(bundle.execute(f"SELECT count(*) FROM {table}").fetchone()[0] != len(tables[table])
               for table in TABLES):
            raise ValueError("bundle contains records outside its declared experiment")
        if _content_hash(tables) != manifest["content_sha256"]:
            raise ValueError("bundle content hash mismatch")
        return bundle, experiment_id, manifest["content_sha256"], tables
    except Exception:
        bundle.close()
        raise


def _insert_exact(connection, table: str, row: dict, primary_key: tuple[str, ...]) -> None:
    columns = tuple(row)
    connection.execute(
        f"INSERT OR IGNORE INTO {_q(table)} ({','.join(_q(c) for c in columns)}) "
        f"VALUES ({','.join('?' for _ in columns)})",
        tuple(row[column] for column in columns),
    )
    where = " AND ".join(f"{_q(column)} IS ?" for column in primary_key)
    saved = connection.execute(
        f"SELECT {','.join(_q(c) for c in columns)} FROM {_q(table)} WHERE {where}",
        tuple(row[column] for column in primary_key),
    ).fetchone()
    if saved is None or tuple(saved) != tuple(row[column] for column in columns):
        raise ValueError(f"{table} identity collision: {tuple(row[column] for column in primary_key)}")


def _insert_agent(connection, row: dict) -> str:
    columns = tuple(row)
    connection.execute(
        f"INSERT OR IGNORE INTO agents ({','.join(_q(c) for c in columns)}) "
        f"VALUES ({','.join('?' for _ in columns)})",
        tuple(row[column] for column in columns),
    )
    saved = connection.execute('SELECT * FROM "agents" WHERE "agent_id"=?', (row["agent_id"],)).fetchone()
    if saved is not None and tuple(saved) == tuple(row[column] for column in columns):
        return row["agent_id"]
    natural = ("label", "algorithm", "checkpoint_sha256", "backend", "rule_set", "config_json")
    condition = " AND ".join(f"{_q(column)} IS ?" for column in natural)
    equivalent = connection.execute(
        f"SELECT agent_id FROM agents WHERE {condition}", tuple(row[column] for column in natural)
    ).fetchone()
    if equivalent is None:
        raise ValueError(f"agents identity collision: {row['agent_id']}")
    return equivalent["agent_id"]


def _insert_surrogate(connection, table: str, row: dict, surrogate: str, identity: tuple[str, ...]) -> None:
    columns = tuple(row)
    columns_without_id = tuple(column for column in columns if column != surrogate)
    condition = " AND ".join(f"{_q(column)} IS ?" for column in identity)
    if table != "mcts_traces":
        saved = connection.execute(
            f"SELECT {','.join(_q(c) for c in columns_without_id)} FROM {_q(table)} WHERE {condition}",
            tuple(row[column] for column in identity),
        ).fetchone()
        if saved is not None:
            if tuple(saved) != tuple(row[column] for column in columns_without_id):
                raise ValueError(f"{table} identity collision")
            return
    connection.execute(
        f"INSERT OR IGNORE INTO {_q(table)} ({','.join(_q(c) for c in columns)}) "
        f"VALUES ({','.join('?' for _ in columns)})",
        tuple(row[column] for column in columns),
    )
    saved = connection.execute(
        f"SELECT {','.join(_q(c) for c in columns)} FROM {_q(table)} WHERE {_q(surrogate)} IS ?",
        (row[surrogate],),
    ).fetchone()
    if saved is not None:
        if tuple(saved) == tuple(row[column] for column in columns):
            return
    elif table != "mcts_traces":
        saved = connection.execute(
            f"SELECT {','.join(_q(c) for c in columns_without_id)} FROM {_q(table)} WHERE {condition}",
            tuple(row[column] for column in identity),
        ).fetchone()
        if saved is not None:
            if tuple(saved) != tuple(row[column] for column in columns_without_id):
                raise ValueError(f"{table} identity collision")
            return
    connection.execute(
        f"INSERT INTO {_q(table)} ({','.join(_q(c) for c in columns_without_id)}) "
        f"VALUES ({','.join('?' for _ in columns_without_id)})",
        tuple(row[column] for column in columns_without_id),
    )


def import_experiment(database: str | Path, bundle_path: str | Path) -> dict[str, Any]:
    """Validate a bundle and add it transactionally; collisions fail without overwriting rows."""
    database_path, package_path = Path(database), Path(bundle_path)
    if database_path.resolve() == package_path.resolve():
        raise ValueError("bundle and destination database must differ")
    bundle, experiment_id, content_hash, tables = _read_bundle(package_path)
    try:
        logging_v2.initialize(database_path, "tournament")
        connection = logging_v2.connect(database_path)
        try:
            identity = f"experiment:{experiment_id}"
            if connection.execute(
                "SELECT 1 FROM source_imports WHERE source_hash=? AND source_identity=?",
                (content_hash, identity),
            ).fetchone():
                return {"imported": False, "experiment_id": experiment_id,
                        "content_sha256": content_hash, "bundle_sha256": _file_hash(package_path)}
            connection.execute("SAVEPOINT experiment_import")
            agent_map = {}
            try:
                for row in tables["experiments"]:
                    _insert_exact(connection, "experiments", dict(row), ("experiment_id",))
                for source_agent in tables["agents"]:
                    agent_map[source_agent["agent_id"]] = _insert_agent(connection, dict(source_agent))
                for row in tables["matches"]:
                    record = dict(row)
                    record["agent_a_id"] = agent_map[record["agent_a_id"]]
                    record["agent_b_id"] = agent_map[record["agent_b_id"]]
                    _insert_exact(connection, "matches", record, ("match_id",))
                for row in tables["games"]:
                    record = dict(row)
                    record["black_agent_id"] = agent_map[record["black_agent_id"]]
                    record["white_agent_id"] = agent_map[record["white_agent_id"]]
                    _insert_exact(connection, "games", record, ("game_id",))
                for table, key in (("moves", ("game_id", "move_index")),
                                   ("mcts_roots", ("game_id", "move_index"))):
                    for row in tables[table]:
                        _insert_exact(connection, table, dict(row), key)
                for row in tables["mcts_traces"]:
                    _insert_surrogate(connection, "mcts_traces", dict(row), "trace_id",
                                      ("game_id", "move_index", "encoding", "data"))
                for row in tables["rating_systems"]:
                    _insert_exact(connection, "rating_systems", dict(row), ("rating_system_id",))
                for row in tables["rating_events"]:
                    record = dict(row)
                    record["agent_id"] = agent_map[record["agent_id"]]
                    _insert_surrogate(connection, "rating_events", record, "rating_event_id",
                                      ("rating_system_id", "sequence", "agent_id"))
                for row in tables["events"]:
                    _insert_surrogate(connection, "events", dict(row), "event_id",
                                      ("experiment_id", "recorded_at", "level", "event", "payload_json"))
                record_count = sum(len(rows) for rows in tables.values())
                connection.execute(
                    "INSERT INTO source_imports(source_hash, source_identity, source_path, imported_at, "
                    "record_count, status, details_json) VALUES (?, ?, ?, ?, ?, 'imported', ?)",
                    (content_hash, identity, str(package_path.resolve()), datetime.now(timezone.utc).isoformat(),
                     record_count, logging_v2.json_dumps({"agent_id_mapping": agent_map})),
                )
                if connection.execute("PRAGMA foreign_key_check").fetchall():
                    raise RuntimeError("destination foreign_key_check failed after import")
                connection.execute("RELEASE experiment_import")
            except Exception:
                connection.execute("ROLLBACK TO experiment_import")
                connection.execute("RELEASE experiment_import")
                raise
            return {"imported": True, "experiment_id": experiment_id,
                    "content_sha256": content_hash, "bundle_sha256": _file_hash(package_path)}
        finally:
            connection.close()
    finally:
        bundle.close()
