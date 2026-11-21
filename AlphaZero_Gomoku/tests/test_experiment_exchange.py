import sqlite3
import pytest

from az.elo import recalculate_elo
from az.experiment_exchange import export_experiment, import_experiment
from az.logging_v2 import (
    connect,
    initialize,
    record_tournament_agent,
    record_tournament_experiment,
    record_tournament_game,
    record_tournament_match,
    write_event,
)


def _create_experiment(connection, experiment_id="exp", config=None):
    record_tournament_experiment(
        connection, experiment_id=experiment_id, started_at="2026-01-01T00:00:00Z",
        backend="python", rule_set="normal", seed=17, config=config or {"budget": "sims:1"},
        metadata={"research_note": "paired baseline"},
    )
    for agent_id in ("a", "b"):
        record_tournament_agent(
            connection, agent_id=f"{experiment_id}:{agent_id}", label=agent_id, algorithm="test",
            checkpoint_sha256=None, backend="python", rule_set="normal", config={"level": 1},
        )
    record_tournament_match(
        connection, match_id=f"{experiment_id}:pair", experiment_id=experiment_id,
        agent_a_id=f"{experiment_id}:a", agent_b_id=f"{experiment_id}:b",
        pair_id="a|b:0", rule_set="normal", seed=None,
    )
    record_tournament_game(
        connection, experiment_id=experiment_id, local_game_id=0,
        match_id=f"{experiment_id}:pair", black_agent_id=f"{experiment_id}:a",
        white_agent_id=f"{experiment_id}:b",
        record={"game_id": 0, "rule_set": "normal", "winner": "black",
                "end_reason": "five_in_row", "moves_count": 1, "duration_sec": 0.1,
                "pair": ["a", "b"], "first_player": "a", "second_player": "b", "pair_game": 0,
                "moves": [60], "raw_moves": [{"player": 1, "simulations_run": 1, "backend": "python"}]},
    )
    write_event(
        connection, experiment_id=experiment_id, event="checkpoint_evaluated",
        payload={"checkpoint_sha256": "abc", "result": "baseline"}, recorded_at="2026-01-01T00:01:00Z",
    )
    connection.execute(
        "INSERT INTO mcts_roots(game_id, move_index, encoding, data) VALUES (?, ?, ?, ?)",
        (f"{experiment_id}:0", 0, "zlib", b"root-data"),
    )


def test_exchange_preserves_research_history_elo_and_is_additive(tmp_path):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    bundle = tmp_path / "experiment.sqlite"
    initialize(source, "tournament")
    with connect(source) as connection:
        _create_experiment(connection)
        recalculate_elo(connection, "exp")

    initialize(target, "tournament")
    with connect(target) as connection:
        _create_experiment(connection, "keep", {"budget": "sims:1"})

    checksum = export_experiment(source, bundle, "exp")
    assert len(checksum) == 64
    result = import_experiment(target, bundle)
    assert result["imported"] is True
    repeated = import_experiment(target, bundle)
    assert repeated["imported"] is False
    with connect(target) as connection:
        assert connection.execute("SELECT count(*) FROM experiments").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM games WHERE experiment_id='exp'").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM events WHERE experiment_id='exp'").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM source_imports").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM games WHERE experiment_id='keep'").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM rating_events").fetchone()[0] == 2
        assert connection.execute("SELECT data FROM mcts_roots").fetchone()[0] == b"root-data"
        assert connection.execute("SELECT config_json FROM experiments WHERE experiment_id='exp'").fetchone()[0] == '{"budget":"sims:1"}'

    reexport = tmp_path / "reexport.sqlite"
    export_experiment(target, reexport, "exp")
    assert import_experiment(source, reexport)["imported"] is True
    with connect(source) as connection:
        assert connection.execute("SELECT count(*) FROM events WHERE experiment_id='exp'").fetchone()[0] == 1


def test_exchange_rejects_existing_paths_and_conflicting_experiments(tmp_path):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    bundle = tmp_path / "experiment.sqlite"
    initialize(source, "tournament")
    with connect(source) as connection:
        _create_experiment(connection)
    export_experiment(source, bundle, "exp")
    with pytest.raises(FileExistsError):
        export_experiment(source, bundle, "exp")

    initialize(target, "tournament")
    with connect(target) as connection:
        record_tournament_experiment(
            connection, experiment_id="exp", started_at="different", backend="python",
            rule_set="normal", seed=17, config={"other": True},
        )
    with pytest.raises(ValueError, match="identity collision"):
        import_experiment(target, bundle)
    with connect(target) as connection:
        assert connection.execute("SELECT config_json FROM experiments WHERE experiment_id='exp'").fetchone()[0] == '{"other":true}'
        assert connection.execute("SELECT count(*) FROM games").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM source_imports").fetchone()[0] == 0


def test_failed_export_does_not_leave_a_database_at_the_requested_path(tmp_path):
    source = tmp_path / "source.sqlite"
    bundle = tmp_path / "missing.sqlite"
    initialize(source, "tournament")
    with pytest.raises(KeyError):
        export_experiment(source, bundle, "missing")
    assert not bundle.exists()


def test_import_rejects_a_tampered_bundle_before_creating_target(tmp_path):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "target.sqlite"
    bundle = tmp_path / "experiment.sqlite"
    initialize(source, "tournament")
    with connect(source) as connection:
        _create_experiment(connection)
    export_experiment(source, bundle, "exp")
    with sqlite3.connect(bundle) as connection:
        connection.execute("UPDATE games SET payload_json='{}'")

    with pytest.raises(ValueError, match="content hash"):
        import_experiment(target, bundle)
    assert not target.exists()


def test_tournament_analytics_reads_sqlite_without_default_exports(tmp_path):
    from scripts.analyze_t35 import analyze

    database = tmp_path / "tournament.sqlite"
    initialize(database, "tournament")
    with connect(database) as connection:
        _create_experiment(connection, config={"budget": "sims:1"})

    result = analyze(database, experiment_ids=["exp"], seed=None)

    assert result["games"] == 1
    pair = result["groups"]["normal_fixed"][0]
    assert (pair["W"], pair["D"], pair["L"]) == (1, 0, 0)
    assert pair["score"] == 1
    assert result["rankings"]["normal_fixed"][0]["agent"] == "a"
    assert not list(tmp_path.glob("*.json"))
