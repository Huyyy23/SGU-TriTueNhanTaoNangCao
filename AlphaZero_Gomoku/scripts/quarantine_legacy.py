"""Preserve ambiguous legacy sources as isolated, lossless SQLite events."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path


def quarantine_source(db: sqlite3.Connection, source: Path, identity: str, reason: str) -> bool:
    """Append one version of a source, or verify an identical version already exists."""
    if not identity or not reason:
        raise ValueError("identity and reason are required")
    before = source.stat()
    raw = source.read_bytes()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"source changed while reading: {source}")
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.dumps({
        "source_identity": identity,
        "source_sha256": digest,
        "reason": reason,
        "encoding": "zlib+base64",
        "data": base64.b64encode(zlib.compress(raw)).decode("ascii"),
    }, sort_keys=True, separators=(",", ":"))
    if db.in_transaction:
        raise ValueError("quarantine requires a connection without an active transaction")
    try:
        db.execute("BEGIN IMMEDIATE")
        rows = db.execute(
            "SELECT payload_json FROM events WHERE event='legacy_quarantine' "
            "AND json_extract(payload_json, '$.source_identity')=? "
            "AND json_extract(payload_json, '$.source_sha256')=?",
            (identity, digest),
        ).fetchall()
        if rows:
            if len(rows) != 1 or zlib.decompress(base64.b64decode(json.loads(rows[0][0])["data"], validate=True)) != raw:
                raise ValueError("conflicting quarantine entry")
            db.commit()
            return False
        db.execute(
            "INSERT INTO events(recorded_at, level, event, payload_json) VALUES (?, 'warning', 'legacy_quarantine', ?)",
            (datetime.now(timezone.utc).isoformat(), payload),
        )
        db.commit()
        return True
    except BaseException:
        db.rollback()
        raise


def restore_source(db: sqlite3.Connection, identity: str, digest: str) -> bytes:
    rows = db.execute(
        "SELECT payload_json FROM events WHERE event='legacy_quarantine' "
        "AND json_extract(payload_json, '$.source_identity')=? "
        "AND json_extract(payload_json, '$.source_sha256')=?",
        (identity, digest),
    ).fetchall()
    if not rows:
        raise KeyError((identity, digest))
    if len(rows) != 1:
        raise ValueError("duplicate quarantine entries")
    payload = json.loads(rows[0][0])
    if payload.get("encoding") != "zlib+base64":
        raise ValueError("unknown quarantine encoding")
    raw = zlib.decompress(base64.b64decode(payload["data"], validate=True))
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("quarantine hash mismatch")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    identity = args.source.resolve().relative_to(root).as_posix()
    with sqlite3.connect(args.db.resolve().as_uri() + "?mode=rw", uri=True, timeout=30) as db:
        inserted = quarantine_source(db, args.source, identity, args.reason)
    print(json.dumps({"source_identity": identity, "status": "quarantined" if inserted else "already_quarantined"}))


if __name__ == "__main__":
    main()
