"""SQLite persistence for backend selection telemetry."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from az.logging_v2 import SELFPLAY_DB, TOURNAMENT_DB, connect, initialize, json_dumps, write_event


def log_backend_selection(
    *,
    run_id: str,
    workload: str,
    requested_backend: str,
    backends: dict,
    components_used: dict,
    primary_path: str | Path | None = None,
) -> None:
    """Record backend selection in the primary SQLite database."""
    recorded_at = datetime.now(timezone.utc).isoformat()

    database = "selfplay" if workload == "selfplay" else "tournament"
    path = Path(primary_path) if primary_path is not None else (
        SELFPLAY_DB if database == "selfplay" else TOURNAMENT_DB
    )
    initialize(path, database)
    payload = {
        "run_id": run_id,
        "workload": workload,
        "requested_backend": requested_backend,
        "backends": backends,
        "components_used": components_used,
    }
    with connect(path) as connection:
        if database == "selfplay":
            exists = connection.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
        else:
            exists = connection.execute("SELECT 1 FROM experiments WHERE experiment_id=?", (run_id,)).fetchone()
        exists = exists is not None
        event_owner = "run_id" if database == "selfplay" else "experiment_id"
        owner_id = run_id if exists else None
        duplicate = connection.execute(
            f"SELECT 1 FROM events WHERE {event_owner} IS ? AND event='backend_selected' AND payload_json=?",
            (owner_id, json_dumps(payload)),
        ).fetchone()
        if duplicate is None:
            if database == "selfplay" and not exists:
                connection.execute(
                    "INSERT INTO events(run_id, recorded_at, level, event, payload_json) "
                    "VALUES(NULL, ?, 'info', 'backend_selected', ?)",
                    (recorded_at, json_dumps(payload)),
                )
            else:
                write_event(
                    connection, event="backend_selected", payload=payload,
                    run_id=run_id if database == "selfplay" else None,
                    experiment_id=run_id if database == "tournament" else None,
                    recorded_at=recorded_at,
                )
