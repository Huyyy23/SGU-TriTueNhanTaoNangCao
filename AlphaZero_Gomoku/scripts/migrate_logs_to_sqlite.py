"""Dry-run inventory and additive migration of historical Gomoku logs to SQLite V2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from az import logging_v2  # noqa: E402

IGNORED_PARTS = {"checkpoint", "checkpoints", "replay", "replays", "archive", "archives", "__pycache__"}
SUPPORTED_SUFFIXES = {".jsonl", ".json", ".csv", ".log", ".txt", ".sqlite", ".sqlite3"}


def _eligible(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES and not any(
        part.casefold() in IGNORED_PARTS for part in path.parts
    ) and not any(part.casefold().endswith((".zip", ".ipynb")) for part in path.parts)


def _records(path: Path, raw: bytes) -> tuple[list[dict], str | None, list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".sqlite", ".sqlite3"}:
        try:
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as source:
                tables = {row[0] for row in source.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )}
                if "backend_runs" not in tables:
                    return [], None, ["unrecognized SQLite schema"]
                cursor = source.execute("SELECT * FROM backend_runs ORDER BY id")
                columns = [column[0] for column in cursor.description]
                return [
                    {"line": index, "record": dict(zip(columns, row))}
                    for index, row in enumerate(cursor.fetchall(), 1)
                ], "backend_selection", []
        except sqlite3.Error as exc:
            return [], None, [str(exc)]
    suffix = path.suffix.lower()
    errors: list[str] = []
    rows: list[dict] = []
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return [], None, [str(exc)]
    if suffix == ".jsonl":
        kind = None
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("record is not an object")
                record_kind = _kind(value)
                if record_kind is None or kind is not None and kind != record_kind:
                    raise ValueError("unknown or mixed record format")
                kind = record_kind
                rows.append({"line": number, "record": value})
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"line {number}: {exc}")
        return rows, kind, errors
    if suffix == ".json":
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return [{"line": 1, "record": value}], _kind(value) or "metadata", []
            if isinstance(value, list) and all(isinstance(row, dict) for row in value):
                kinds = {_kind(row) for row in value}
                if len(kinds) == 1 and None not in kinds:
                    return [{"line": i, "record": row} for i, row in enumerate(value, 1)], kinds.pop(), []
                return [], None, ["JSON array has unknown or mixed record types"]
            return [], None, ["JSON root is not an object or record array"]
        except (ValueError, json.JSONDecodeError) as exc:
            return [], None, [str(exc)]
    if suffix == ".csv":
        try:
            reader = csv.DictReader(text.splitlines())
            if not reader.fieldnames:
                return [], None, ["CSV has no header"]
            return [{"line": i, "record": dict(row)} for i, row in enumerate(reader, 2)], "csv", []
        except csv.Error as exc:
            return [], None, [str(exc)]
    return [{"line": 1, "record": {"raw_text": text}}], "opaque", []



def _kind(row: dict) -> str | None:
    if {"run_id", "workload", "requested_backend", "mcts_backend", "patterns_backend"} <= row.keys():
        return "backend_selection"
    if "winner" in row and "moves_count" in row and "pair" in row and "run_id" in row:
        return "tournament_game"
    if "global_step" in row and "iteration" in row:
        return "training"
    if "iteration" in row and {"loss", "policy_loss", "value_loss"} & row.keys():
        return "training_metric"
    if ("moves_s" in row or "moves_per_sec" in row or "games_per_hour" in row) and (
        "seed" in row or "parallel_games" in row or "workers" in row
    ):
        return "benchmark"
    if {"game_id", "rule_set", "winner", "end_reason", "duration_sec"} <= row.keys() and (
        "moves_count" in row or "num_moves" in row
    ):
        return "game"
    return None

def _family(file: dict) -> str | None:
    if file["kind"] == "tournament_game":
        return "tournament"
    if file["kind"] == "backend_selection":
        workloads = {item["record"].get("workload") for item in file["records"]}
        if workloads == {"selfplay"}:
            return "selfplay"
        if workloads == {"tournament"}:
            return "tournament"
        return None
    if file["kind"] in {"metadata", "csv"} and file["records"]:
        row = file["records"][0]["record"]
        keys = set(row)
        config = row.get("config")
        if isinstance(config, dict):
            keys |= set(config)
        if keys & {
            "agent", "agents", "pair", "pairs", "pairs_summary", "matrix", "rankings",
            "match_id", "win_rate", "score", "config_a", "config_b",
        }:
            return "tournament"
    return "selfplay"


def inventory(paths: list[Path]) -> list[dict]:
    """Read source bytes only; callers pass explicit paths or recursively found candidates."""
    result = []
    for path in paths:
        if not _eligible(path):
            continue
        raw = path.read_bytes()
        digest = hashlib.sha256(raw)
        if path.suffix.lower() in {".sqlite", ".sqlite3"}:
            wal = path.with_name(path.name + "-wal")
            if wal.is_file():
                digest.update(wal.read_bytes())
        records, kind, errors = _records(path, raw)
        resolved = path.resolve()
        try:
            identity = resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            identity = str(resolved)
        status = "ready" if records and not errors else "skipped"
        if kind == "training" and records:
            runs = {item["record"].get("run_id") for item in records}
            status = "ready" if len(runs) == 1 and not errors else "skipped"
        if kind == "game" and records:
            runs = {item["record"].get("run_id") for item in records}
            status = "ready" if len(runs) == 1 and all(
                _game_ok(item["record"]) for item in records
            ) and not errors else "skipped"
        if kind == "tournament_game" and records:
            identities = {(item["record"].get("run_id"), item["record"].get("rule_set")) for item in records}
            status = "ready" if len(identities) == 1 and all(
                _tournament_ok(item["record"]) for item in records
            ) and not errors else "skipped"
        result.append({"path": identity, "identity": identity, "hash": digest.hexdigest(),
                       "kind": kind, "records": records, "status": status, "errors": errors})
    return result


def _game_ok(row: dict) -> bool:
    return (
        row.get("rule_set") in {"normal", "renju"}
        and row.get("winner") in {"black", "white", "draw"}
        and isinstance(row.get("end_reason"), str)
        and isinstance(row.get("duration_sec"), (int, float))
        and isinstance(row.get("moves_count", row.get("num_moves")), int)
    )


def _tournament_ok(row: dict) -> bool:
    pair = row.get("pair")
    players = (row.get("first_player"), row.get("second_player"))
    moves = row.get("moves") or []
    return (
        _game_ok(row)
        and isinstance(pair, list)
        and len(pair) == 2
        and all(isinstance(value, str) and value for value in pair)
        and all(player in pair for player in players)
        and row.get("run_id") is not None
        and row.get("game_id") is not None
        and (not moves or not isinstance(moves[0], dict)
             or all(isinstance(move, dict) and {"row", "col"} <= move.keys() for move in moves))
    )


def _source_event(
    db, file: dict, item: dict, *, run_id: str | None = None, experiment_id: str | None = None
) -> None:
    record = item["record"]
    payload = {
        "source_path": file["path"],
        "source_hash": file["hash"],
        "source_identity": file["identity"],
        "kind": file["kind"],
        "line": item["line"],
        "record_hash": hashlib.sha256(logging_v2.json_dumps(record).encode()).hexdigest(),
        "raw_record": record,
    }
    recorded_at = record.get("recorded_at") or record.get("timestamp") or "legacy-unknown"
    if run_id is not None or experiment_id is not None:
        logging_v2.write_event(
            db, event=f"historical_{file['kind']}_record", payload=payload,
            run_id=run_id, experiment_id=experiment_id, recorded_at=recorded_at,
        )
    elif file["family"] == "selfplay":
        db.execute(
            "INSERT INTO events(run_id, recorded_at, level, event, payload_json) VALUES(NULL, ?, 'info', ?, ?)",
            (recorded_at, f"historical_{file['kind']}_record", logging_v2.json_dumps(payload)),
        )
    else:
        logging_v2.write_event(
            db, event=f"historical_{file['kind']}_record", payload=payload,
            experiment_id=None, recorded_at=recorded_at,
        )


def _run_identity(file: dict, source_run_id: str | None = None) -> str:
    return "legacy:" + hashlib.sha256(f"{file['identity']}\0{source_run_id or ''}".encode()).hexdigest()




def _prepare(db_path: Path, family: str) -> None:
    logging_v2.initialize(db_path, family)


def _legacy_timestamp(record: dict) -> str:
    return record.get("timestamp") or record.get("recorded_at") or "legacy-unknown"


def _legacy_backend(record: dict) -> str:
    value = record.get("backend") or record.get("mcts_backend") or "legacy-unknown"
    return value if isinstance(value, str) else logging_v2.json_dumps(value)


def _prepare_selfplay_run(db, file: dict, records: list[dict]) -> str:
    first = records[0]
    original_run_id = first.get("run_id")
    run_id = _run_identity(file, str(original_run_id) if original_run_id is not None else None)
    config = first.get("config") if isinstance(first.get("config"), dict) else {}
    ruleset = first.get("rule_set") or "unknown"
    seed = first.get("seed")
    logging_v2.record_selfplay_run(
        db,
        run_id=run_id,
        started_at=_legacy_timestamp(first),
        backend=_legacy_backend(first),
        rule_set=ruleset,
        seed=seed if isinstance(seed, int) else None,
        config=config,
        metadata={
            "legacy_source_identity": file["identity"],
            "legacy_run_id": original_run_id,
        },
    )
    return run_id


def _prepare_tournament_experiment(db, file: dict, records: list[dict]) -> tuple[str, dict]:
    first = records[0]
    original_id = str(first["run_id"])
    experiment_id = _run_identity(file, original_id)
    config = first.get("config") if isinstance(first.get("config"), dict) else {}
    ruleset = first["rule_set"]
    logging_v2.record_tournament_experiment(
        db, experiment_id=experiment_id, started_at=_legacy_timestamp(first),
        backend=_legacy_backend(first), rule_set=ruleset,
        seed=first.get("seed") if isinstance(first.get("seed"), int) else None,
        config=config,
        metadata={"legacy_source_identity": file["identity"], "legacy_run_id": original_id},
    )
    labels = sorted({label for record in records for label in record["pair"]})
    settings = first.get("agent_settings") if isinstance(first.get("agent_settings"), dict) else {}
    agent_ids = {}
    for label in labels:
        agent_config = settings.get(label, {})
        algorithm = agent_config.get("algorithm", label) if isinstance(agent_config, dict) else label
        agent_backend = agent_config.get("backend") if isinstance(agent_config, dict) else None
        agent_id = logging_v2.tournament_agent_id(
            label=label, algorithm=algorithm, checkpoint_sha256=None, backend=agent_backend,
            rule_set=ruleset, config=agent_config if isinstance(agent_config, dict) else {},
        )
        logging_v2.record_tournament_agent(
            db, agent_id=agent_id, label=label, algorithm=algorithm,
            checkpoint_sha256=None, backend=agent_backend, rule_set=ruleset,
            config=agent_config if isinstance(agent_config, dict) else {},
        )
        agent_ids[label] = agent_id
    match_ids = {}
    for pair in sorted({tuple(record["pair"]) for record in records}):
        pair_id = "|".join(pair)
        match_id = f"{experiment_id}:{pair_id}"
        logging_v2.record_tournament_match(
            db, match_id=match_id, experiment_id=experiment_id,
            agent_a_id=agent_ids[pair[0]], agent_b_id=agent_ids[pair[1]],
            pair_id=pair_id, rule_set=ruleset, seed=None,
            metadata={"legacy_source_identity": file["identity"]},
        )
        match_ids[pair_id] = match_id
    return experiment_id, {"agents": agent_ids, "matches": match_ids}


def _import_normalized(db, file: dict, item: dict, owner: dict) -> None:
    record = item["record"]
    if file["kind"] == "game":
        normalized = {key: value for key, value in record.items() if key != "root_data"}
        normalized["game_id"] = str(record["game_id"])
        normalized["moves_count"] = record.get("moves_count", record.get("num_moves"))
        normalized["moves"] = record.get("moves") or []
        logging_v2.record_selfplay_game(db, owner["id"], normalized)
    elif file["kind"] == "training":
        logging_v2.record_training_step(
            db, run_id=owner["id"], iteration=int(record["iteration"]),
            global_step=int(record["global_step"]), recorded_at=_legacy_timestamp(record),
            metrics=record,
        )
    elif file["kind"] == "tournament_game":
        raw_moves = record.get("raw_moves")
        moves = record.get("moves") or []
        if moves and isinstance(moves[0], dict):
            raw_moves = raw_moves or moves
            moves = [int(move["row"]) * 11 + int(move["col"]) for move in moves]
        normalized = {**record, "moves": moves, "raw_moves": raw_moves or []}
        pair_id = "|".join(record["pair"])
        logging_v2.record_tournament_game(
            db, experiment_id=owner["id"], local_game_id=record["game_id"],
            match_id=owner["matches"][pair_id],
            black_agent_id=owner["agents"][record["first_player"]],
            white_agent_id=owner["agents"][record["second_player"]], record=normalized,
        )


def import_files(files: list[dict], db_path: Path, tournament_db_path: Path | None = None) -> tuple[int, int]:
    """Add normalized records and provenance events; never rewrite source/database rows."""
    pending = [file for file in files if file["status"] == "ready"]
    skipped = len(files) - len(pending)
    expanded = []
    for source in pending:
        source.setdefault("_outcomes", [])
        if source["kind"] == "backend_selection":
            partitions = {}
            for item in source["records"]:
                workload = item["record"].get("workload")
                family = {"selfplay": "selfplay", "tournament": "tournament"}.get(workload)
                if family is not None:
                    partitions.setdefault(family, []).append(item)
            if not partitions:
                source["status"] = "skipped"
                source["errors"].append("backend database has no recognized workloads")
                skipped += 1
                continue
            if sum(len(items) for items in partitions.values()) != len(source["records"]):
                source["errors"].append("some backend rows have unknown workloads")
                source["_outcomes"].append("skipped")
                skipped += 1
            for family, records in partitions.items():
                expanded.append({**source, "records": records, "family": family, "_source": source})
            continue
        family = _family(source)
        if family is None:
            source["status"] = "skipped"
            source["errors"].append("source contains mixed or unknown workloads")
            skipped += 1
            continue
        if family == "tournament" and tournament_db_path is None:
            source["status"] = "skipped"
            source["errors"].append("tournament target database was not provided")
            skipped += 1
            continue
        expanded.append({**source, "family": family, "_source": source})
    outcomes = {}

    targets = {"selfplay": Path(db_path)}
    if tournament_db_path is not None:
        targets["tournament"] = Path(tournament_db_path)
    if expanded:
        database_paths = [path.resolve() for path in targets.values()]
        if len(set(database_paths)) != len(database_paths):
            raise ValueError("self-play and tournament databases must be different files")
        for family, target in targets.items():
            logging_v2.initialize(target, family)
    imported = 0
    for file in expanded:
        family = file["family"]
        target = targets[family]
        logging_v2.initialize(target, family)
        db = logging_v2.connect(target)
        try:
            with db:
                if db.execute(
                    "SELECT 1 FROM source_imports WHERE source_hash=? AND source_identity=?",
                    (file["hash"], file["identity"]),
                ).fetchone():
                    file["status"] = "already_imported"
                    file["_source"]["_outcomes"].append(file["status"])
                    skipped += 1
                    continue
                db.execute("SAVEPOINT historical_source")
                try:
                    records = [item["record"] for item in file["records"]]
                    owner = {}
                    if family == "selfplay" and file["kind"] in {"game", "training"}:
                        owner["id"] = _prepare_selfplay_run(db, file, records)
                    elif family == "tournament" and file["kind"] == "tournament_game":
                        owner["id"], tournament_owner = _prepare_tournament_experiment(db, file, records)
                        owner.update(tournament_owner)
                    for item in file["records"]:
                        _import_normalized(db, file, item, owner)
                        _source_event(
                            db, file, item,
                            run_id=owner.get("id") if family == "selfplay" else None,
                            experiment_id=owner.get("id") if family == "tournament" else None,
                        )
                    db.execute(
                        "INSERT INTO source_imports(source_hash,source_identity,source_path,imported_at,"
                        "record_count,status,details_json) VALUES(?,?,?,?,?,?,?)",
                        (file["hash"], file["identity"], file["path"],
                         datetime.now(timezone.utc).isoformat(), len(file["records"]), "imported",
                         logging_v2.json_dumps({"kind": file["kind"], "family": family})),
                    )
                    db.execute("RELEASE historical_source")
                    file["status"] = "imported"
                    file["_source"]["_outcomes"].append(file["status"])
                    imported += 1
                except Exception as exc:
                    db.execute("ROLLBACK TO historical_source")
                    db.execute("RELEASE historical_source")
                    file["status"] = "skipped"
                    file["errors"].append(str(exc))
                    file["_source"]["_outcomes"].append(file["status"])
                    skipped += 1
        finally:
            db.close()
    for source in files:
        outcomes = source.pop("_outcomes", [])
        if outcomes:
            if all(status in {"imported", "already_imported"} for status in outcomes):
                source["status"] = "imported" if "imported" in outcomes else "already_imported"
            elif any(status in {"imported", "already_imported"} for status in outcomes):
                source["status"] = "partial"
            else:
                source["status"] = "skipped"
    return imported, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, help="additional source files/directories")
    parser.add_argument("--import", dest="do_import", action="store_true", help="write additive imports; default is dry-run")
    parser.add_argument("--db", type=Path, default=ROOT / "data/selfplay.sqlite")
    parser.add_argument("--tournament-db", type=Path, default=ROOT / "data/tournament.sqlite")
    args = parser.parse_args(argv)
    candidates: list[Path] = []
    for root in (ROOT / "logs", ROOT / "evidence", *args.paths):
        if root.is_file():
            candidates.append(root)
        elif root.is_dir():
            candidates.extend(sorted(path for path in root.rglob("*") if _eligible(path)))
    unique = {str(path.resolve()): path for path in candidates}
    files = inventory(list(unique.values()))
    if args.do_import:
        imported, skipped = import_files(files, args.db, args.tournament_db)
    else:
        imported, skipped = 0, sum(file["status"] != "ready" for file in files)
    for file in files:
        print(json.dumps({key: file[key] for key in ("path", "hash", "kind", "status", "errors")}, ensure_ascii=False))
    print(json.dumps({"sources": len(files), "imported": imported, "skipped": skipped, "mode": "import" if args.do_import else "dry-run"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

