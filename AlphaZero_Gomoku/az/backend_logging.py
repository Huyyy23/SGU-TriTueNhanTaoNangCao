"""SQLite index for the actual backends selected by training and tournaments."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3


def log_backend_selection(
    *,
    run_id: str,
    workload: str,
    requested_backend: str,
    backends: dict,
    components_used: dict,
    db_path: str | Path = "logs/backend_runs.sqlite3",
) -> None:
    """Append one parent-process backend-resolution record per run invocation."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=30) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS backend_runs ("
            "id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, workload TEXT NOT NULL, "
            "requested_backend TEXT NOT NULL, mcts_backend TEXT NOT NULL, patterns_backend TEXT NOT NULL, "
            "components_used TEXT NOT NULL, compiler TEXT, native_mcts_version TEXT, "
            "native_patterns_version TEXT, recorded_at_utc TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO backend_runs (run_id, workload, requested_backend, mcts_backend, patterns_backend, "
            "components_used, compiler, native_mcts_version, native_patterns_version, recorded_at_utc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                workload,
                requested_backend,
                backends["mcts"],
                backends["patterns"],
                json.dumps(components_used, sort_keys=True, separators=(",", ":")),
                backends.get("compiler"),
                backends.get("native_mcts_version"),
                backends.get("native_patterns_version"),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
