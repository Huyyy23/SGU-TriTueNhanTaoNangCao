import json

from scripts.migrate_logs_to_sqlite import import_files, inventory, main
from az.logging_v2 import connect


def test_inventory_preserves_hash_and_skips_unknown(tmp_path):
    good = tmp_path / "games.jsonl"
    good.write_text(json.dumps({"run_id": "r", "game_id": 1, "rule_set": "normal", "winner": "draw",
                                "end_reason": "draw", "moves_count": 0, "duration_sec": 0, "moves": []}) + "\n")
    unknown = tmp_path / "other.jsonl"
    unknown.write_text('{"anything": 1}\n')
    found = inventory([good, unknown])
    assert found[0]["status"] == "ready" and found[0]["hash"]
    assert found[1]["status"] == "skipped"


def test_import_is_transactional_and_source_identity_idempotent(tmp_path):
    source = tmp_path / "games.jsonl"
    source.write_text(json.dumps({"run_id": "r", "game_id": 1, "rule_set": "normal", "winner": "draw",
                                  "end_reason": "draw", "moves_count": 0, "duration_sec": 0, "moves": []}) + "\n")
    files = inventory([source])
    db = tmp_path / "db.sqlite"
    assert import_files(files, db) == (1, 0)
    assert import_files(files, db) == (0, 1)
    with connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM source_imports").fetchone()[0] == 1

def test_identical_files_from_distinct_sources_remain_distinct(tmp_path):
    row = {"run_id": "r", "game_id": 1, "rule_set": "normal", "winner": "draw",
           "end_reason": "draw", "moves_count": 0, "duration_sec": 0, "moves": []}
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    content = json.dumps(row) + "\n"
    first.write_text(content)
    second.write_text(content)
    files = inventory([first, second])
    assert files[0]["hash"] == files[1]["hash"]

    db = tmp_path / "db.sqlite"
    assert import_files(files, db) == (2, 0)
    with connect(db) as connection:
        assert connection.execute("SELECT count(*) FROM games").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM source_imports").fetchone()[0] == 2


def test_dry_run_defaults_to_roots_without_writing(tmp_path, monkeypatch, capsys):
    logs, evidence = tmp_path / "logs", tmp_path / "evidence"
    logs.mkdir()
    evidence.mkdir()
    source = logs / "opaque.log"
    source.write_text("console output\n")
    monkeypatch.setattr("scripts.migrate_logs_to_sqlite.ROOT", tmp_path)
    assert main([]) == 0
    assert not (tmp_path / "data" / "selfplay.sqlite").exists()
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["mode"] == "dry-run"


def test_opaque_text_is_preserved_as_an_exact_source_event(tmp_path):
    source = tmp_path / "console.txt"
    original = "not a parser target\n"
    source.write_text(original)
    before = source.read_bytes()
    database = tmp_path / "selfplay.sqlite"
    files = inventory([source])
    assert files[0]["kind"] == "opaque"
    assert import_files(files, database) == (1, 0)
    assert source.read_bytes() == before
    with connect(database) as db:
        event = json.loads(db.execute("SELECT payload_json FROM events").fetchone()[0])
        assert event["raw_record"]["raw_text"] == before.decode("utf-8-sig")


def test_training_record_and_benchmark_are_preserved_as_events(tmp_path):
    training = tmp_path / "training.jsonl"
    training.write_text(json.dumps({"run_id": "r1", "iteration": 2, "global_step": 10, "loss": 1.2}) + "\n")
    benchmark = tmp_path / "bench.jsonl"
    benchmark.write_text(json.dumps({"mode": "fixed", "seed": 0, "moves_s": 12.5}) + "\n")
    files = inventory([training, benchmark])
    assert [f["kind"] for f in files] == ["training", "benchmark"]
    database = tmp_path / "selfplay.sqlite"
    assert import_files(files, database) == (2, 0)
    with connect(database) as db:
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 1


def test_training_without_embedded_run_id_uses_source_identity(tmp_path):
    training = tmp_path / "training.jsonl"
    training.write_text(json.dumps({"iteration": 1, "global_step": 8, "loss": 0.7}) + "\n")
    database = tmp_path / "selfplay.sqlite"
    files = inventory([training])

    assert files[0]["status"] == "ready"
    assert import_files(files, database) == (1, 0)
    with connect(database) as db:
        run_id, = db.execute("SELECT run_id FROM training_steps").fetchone()
        assert run_id.startswith("legacy:")
        assert db.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 1
def test_benchmark_json_array_is_preserved_as_events(tmp_path):
    source = tmp_path / "bench_batching.json"
    source.write_text(json.dumps([
        {"parallel_games": workers, "games": 16, "wall_sec": elapsed, "games_per_hour": rate}
        for workers, elapsed, rate in ((1, 268.25, 214.7), (4, 168.43, 342.0))
    ]))
    files = inventory([source])
    assert files[0]["kind"] == "benchmark"
    database = tmp_path / "selfplay.sqlite"
    assert import_files(files, database) == (1, 0)
    with connect(database) as db:
        assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM source_imports").fetchone()[0] == 1



def test_tournament_routing_and_missing_identity_ambiguity(tmp_path):
    tournament = tmp_path / "games.jsonl"
    row = {"run_id": "experiment-1", "game_id": 3, "pair": ["alpha", "beta"],
           "first_player": "alpha", "second_player": "beta", "rule_set": "normal",
           "winner": "black", "end_reason": "five", "moves_count": 1,
           "duration_sec": 0.2, "moves": [12], "pair_game": 0}
    tournament.write_text(json.dumps(row) + "\n")
    selfplay_db, tournament_db = tmp_path / "selfplay.sqlite", tmp_path / "tournament.sqlite"
    files = inventory([tournament])
    assert files[0]["kind"] == "tournament_game"
    assert import_files(files, selfplay_db, tournament_db) == (1, 0)
    with connect(selfplay_db) as db:
        assert db.execute("SELECT count(*) FROM games").fetchone()[0] == 0
    with connect(tournament_db) as db:
        assert db.execute("SELECT count(*) FROM games").fetchone()[0] == 1
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_integrity_check_after_selfplay_import(tmp_path):
    source = tmp_path / "game.jsonl"
    source.write_text(json.dumps({"game_id": 1, "rule_set": "normal", "winner": "draw",
                                  "end_reason": "draw", "num_moves": 0,
                                  "duration_sec": 0}) + "\n")
    database = tmp_path / "selfplay.sqlite"
    assert import_files(inventory([source]), database) == (1, 0)
    with connect(database) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_legacy_backend_sqlite_imports_into_both_primary_databases(tmp_path):
    import sqlite3

    source = tmp_path / "backend_runs.sqlite3"
    with sqlite3.connect(source) as db:
        db.execute(
            "CREATE TABLE backend_runs (id INTEGER PRIMARY KEY, run_id TEXT, workload TEXT, "
            "requested_backend TEXT, mcts_backend TEXT, patterns_backend TEXT, components_used TEXT, "
            "compiler TEXT, native_mcts_version TEXT, native_patterns_version TEXT, recorded_at_utc TEXT)"
        )
        for run_id, workload in (("s1", "selfplay"), ("t1", "tournament")):
            db.execute(
                "INSERT INTO backend_runs VALUES(NULL, ?, ?, 'auto', 'python', 'python', '{}', "
                "NULL, NULL, NULL, 'legacy-unknown')",
                (run_id, workload),
            )
    before = source.read_bytes()
    files = inventory([source])
    assert files[0]["kind"] == "backend_selection"
    selfplay_db, tournament_db = tmp_path / "selfplay.sqlite", tmp_path / "tournament.sqlite"
    assert import_files(files, selfplay_db, tournament_db) == (2, 0)
    assert source.read_bytes() == before
    for path, run_id in ((selfplay_db, "s1"), (tournament_db, "t1")):
        with connect(path) as db:
            payload, = db.execute(
                "SELECT payload_json FROM events WHERE event='historical_backend_selection_record'"
            ).fetchone()
            assert json.loads(payload)["raw_record"]["run_id"] == run_id
            assert db.execute("SELECT count(*) FROM source_imports").fetchone()[0] == 1
    assert import_files(inventory([source]), selfplay_db, tournament_db) == (0, 2)
