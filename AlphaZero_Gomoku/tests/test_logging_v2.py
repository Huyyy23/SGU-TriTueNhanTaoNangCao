import numpy as np
import sqlite3

import pytest

from az.logging_v2 import (
    backup_database,
    connect,
    decode_blob,
    encode_blob,
    initialize,
    record_selfplay_game,
    record_selfplay_run,
    record_training_step,
    tournament_agent_id,
    encode_mcts_root,
    decode_mcts_root,
)
from az.backend_logging import log_backend_selection



@pytest.mark.parametrize(
    ("database", "tables"),
    [
        ("selfplay", {"runs", "games", "moves", "mcts_roots", "mcts_traces", "training_steps", "artifacts", "events", "source_imports", "schema_migrations"}),
        ("tournament", {"experiments", "agents", "matches", "games", "moves", "mcts_roots", "mcts_traces", "rating_systems", "rating_events", "events", "source_imports", "schema_migrations"}),
    ],
)
def test_initialize_is_idempotent_and_enables_referential_integrity(tmp_path, database, tables):
    path = tmp_path / f"{database}.sqlite"
    initialize(path, database)
    initialize(path, database)

    with connect(path) as connection:
        actual = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert tables <= actual
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert [row[0] for row in connection.execute("SELECT version FROM schema_migrations")] == [1]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO games(game_id, run_id, local_game_id, rule_set, winner, end_reason, moves_count, duration_sec, payload_json) "
                "VALUES ('g', 'missing', '0', 'normal', 'draw', 'board_full', 0, 0, '{}')"
            ) if database == "selfplay" else connection.execute(
                "INSERT INTO games(game_id, experiment_id, local_game_id, black_agent_id, white_agent_id, rule_set, winner, end_reason, moves_count, duration_sec, payload_json) "
                "VALUES ('g', 'missing', '0', 'black', 'white', 'normal', 'draw', 'board_full', 0, 0, '{}')"
            )


def test_initialize_existing_database_with_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initialize("relative.sqlite", "selfplay")
    initialize("relative.sqlite", "selfplay")
    with connect("relative.sqlite") as connection:
        assert connection.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 1


def test_initialize_refuses_newer_schema_before_creating_other_tables(tmp_path):
    path = tmp_path / "future.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_migrations VALUES (99, 'future')")

    with pytest.raises(RuntimeError, match="newer"):
        initialize(path, "selfplay")
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == [("schema_migrations",)]


def test_compressed_blob_round_trip_is_lossless():
    raw = bytes(range(256)) * 32
    encoding, compressed = encode_blob(raw)
    assert encoding == "zlib"
    assert decode_blob(encoding, compressed) == raw
    assert decode_blob(*encode_blob(raw, compress=False)) == raw


def test_selfplay_records_are_idempotent_and_collisions_fail(tmp_path):
    path = tmp_path / "selfplay.sqlite"
    initialize(path, "selfplay")
    with connect(path) as connection:
        record_selfplay_run(
            connection, run_id="r1", started_at="2026-01-01T00:00:00Z",
            backend="python", rule_set="normal", seed=3, config={"simulations": 10},
        )
        game = {
            "game_id": 7, "rule_set": "normal", "winner": "draw", "end_reason": "board_full",
            "moves_count": 1, "duration_sec": 0.5,
            "moves": [{"move_index": 0, "player": 1, "row": 5, "col": 5, "value_before": 0.2}],
        }
        assert record_selfplay_game(connection, "r1", game) == "r1:7"
        record_selfplay_game(connection, "r1", game)
        metrics = {"loss": 1.25}
        record_training_step(
            connection, run_id="r1", iteration=1, global_step=5,
            recorded_at="2026-01-01T00:01:00Z", metrics=metrics,
        )
        record_training_step(
            connection, run_id="r1", iteration=1, global_step=5,
            recorded_at="2026-01-01T00:01:00Z", metrics=metrics,
        )
        assert connection.execute("SELECT count(*) FROM games").fetchone()[0] == 1
        assert connection.execute("SELECT action FROM moves").fetchone()[0] == 60
        assert connection.execute("SELECT count(*) FROM training_steps").fetchone()[0] == 1
        with pytest.raises(ValueError, match="collision"):
            record_selfplay_game(connection, "r1", {**game, "winner": "black"})


def test_selfplay_provenance_id_is_database_game_id_after_local_offset(tmp_path):
    path = tmp_path / "selfplay.sqlite"
    initialize(path, "selfplay")
    with connect(path) as connection:
        record_selfplay_run(connection, run_id="run", started_at="start", backend="python",
                            rule_set="normal", seed=1, config={})
        game = {"game_id": 125, "rule_set": "normal", "winner": "draw",
                "end_reason": "board_full", "moves_count": 0, "duration_sec": 0,
                "config": {"provenance": {"game_id": "a" * 64, "game_seed": 123}}}
        assert record_selfplay_game(connection, "run", game) == "a" * 64
        row = connection.execute("SELECT game_id, local_game_id, payload_json FROM games").fetchone()
        assert row[0] == "a" * 64 and row[1] == "125" and '"game_seed":123' in row[2]



def test_selfplay_root_blob_round_trips_and_is_idempotent(tmp_path):
    path = tmp_path / "selfplay.sqlite"
    initialize(path, "selfplay")
    counts = np.zeros(121, dtype=np.int32)
    counts[60] = 3
    root_policy = counts.astype(np.float64) / 3
    target_policy = root_policy.astype(np.float32)
    priors = np.linspace(0.0, 1.0, 121, dtype=np.float64)
    q_values = np.linspace(-1.0, 1.0, 121, dtype=np.float64)
    legal = counts > 0
    forbidden = np.zeros(121, dtype=bool)
    forbidden[0] = True
    encoding, blob = encode_mcts_root(
        counts, root_policy, target_policy, priors, q_values, legal, forbidden
    )

    with connect(path) as connection:
        record_selfplay_run(
            connection, run_id="r", started_at="start", backend="python",
            rule_set="normal", seed=4, config={},
        )
        game = {
            "game_id": 0, "rule_set": "normal", "winner": "draw", "end_reason": "board_full",
            "moves_count": 1, "duration_sec": 0.1,
            "moves": [{"move_index": 0, "player": 1, "row": 5, "col": 5, "value_before": 0.25}],
            "root_data": [{"move_index": 0, "encoding": encoding, "data": blob}],
        }
        record_selfplay_game(connection, "r", game)
        record_selfplay_game(connection, "r", game)
        saved = connection.execute("SELECT encoding, data FROM mcts_roots").fetchone()
        assert tuple(saved) == (encoding, blob)

    decoded = decode_mcts_root(*saved)
    np.testing.assert_array_equal(decoded["visit_counts"], counts)
    np.testing.assert_array_equal(decoded["root_policy"], root_policy)
    np.testing.assert_array_equal(decoded["target_policy"], target_policy)
    np.testing.assert_array_equal(decoded["priors"], priors)
    np.testing.assert_array_equal(decoded["q_values"], q_values)
    np.testing.assert_array_equal(decoded["legal_mask"], legal)
    np.testing.assert_array_equal(decoded["forbidden_mask"], forbidden)

def test_backend_selection_uses_the_primary_database(tmp_path):
    path = tmp_path / "selfplay.sqlite"
    initialize(path, "selfplay")
    with connect(path) as connection:
        record_selfplay_run(
            connection, run_id="r1", started_at="start", backend="cpp",
            rule_set="normal", seed=7, config={},
        )
    log_backend_selection(
        run_id="r1", workload="selfplay", requested_backend="auto",
        backends={"mcts": "cpp", "patterns": "python", "compiler": "clang"},
        components_used={"mcts": "cpp", "patterns": "not_used"},
        primary_path=path,
    )
    with connect(path) as connection:
        row = connection.execute(
            "SELECT run_id, event, payload_json FROM events"
        ).fetchone()
    assert row["run_id"] == "r1"
    assert row["event"] == "backend_selected"
    assert '"requested_backend":"auto"' in row["payload_json"]


def test_online_backup_restores_committed_records(tmp_path):
    source = tmp_path / "source.sqlite"
    backup = tmp_path / "backup.sqlite"
    initialize(source, "selfplay")
    with connect(source) as connection:
        record_selfplay_run(
            connection, run_id="saved", started_at="start", backend="python",
            rule_set="normal", seed=11, config={"simulations": 200},
        )

    backup_database(source, backup)
    with connect(backup) as connection:
        assert connection.execute("SELECT run_id FROM runs").fetchone()[0] == "saved"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_tournament_agent_identity_is_stable_for_the_same_checkpoint():
    identity = {
        "label": "az",
        "algorithm": "az_v0",
        "checkpoint_sha256": "a" * 64,
        "backend": "python",
        "rule_set": "normal",
        "config": {"budget": "sims:200"},
    }
    first = tournament_agent_id(**identity)
    assert tournament_agent_id(**identity) == first
    assert tournament_agent_id(**{**identity, "checkpoint_sha256": "b" * 64}) != first
