import json
import sqlite3

from az.model import AZNet, save_checkpoint
from gomoku.board import Board
from gomoku.rules.normal import NormalRules
import scripts.arena as arena
from scripts.arena import main, make_player


def test_same_checkpoint_with_separate_tactical_configs(tmp_path):
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(AZNet(n_blocks=1, n_filters=4), checkpoint)
    board = Board()
    for move in [(5, 1), (0, 0), (5, 2), (0, 2), (5, 3), (0, 4), (5, 4), (1, 0)]:
        board.play(*move)
    player = make_player(str(checkpoint), NormalRules(), {"tactical": {"enabled": True}}, seed=0)
    assert player.select_move(board) in {(5, 0), (5, 5)}
    assert player.last_override_reason == "win_now"

    base = tmp_path / "base.yaml"
    tactical = tmp_path / "tactical.yaml"
    base.write_text("n_simulations: 1\n", encoding="utf-8")
    tactical.write_text("n_simulations: 1\ntactical:\n  enabled: true\n", encoding="utf-8")
    db_path = tmp_path / "arena.sqlite"
    assert main(["--a", str(checkpoint), "--b", str(checkpoint), "--games", "1",
                 "--config-a", str(base), "--config-b", str(tactical),
                 "--db", str(db_path)]) == 0
    with sqlite3.connect(db_path) as connection:
        summary_event = json.loads(connection.execute(
            "SELECT payload_json FROM events WHERE event='arena_summary'"
        ).fetchone()[0])
        warning_event = json.loads(connection.execute(
            "SELECT payload_json FROM events WHERE event='arena_warnings'"
        ).fetchone()[0])
        game_row = connection.execute(
            "SELECT black_agent_id, white_agent_id, payload_json FROM games"
        ).fetchone()
    record = json.loads(game_row[2])
    summary = summary_event["summary"]
    assert summary["total_games"] == 1
    assert summary["config_a"].get("tactical", {}).get("enabled", False) is False
    assert summary["config_b"]["tactical"]["enabled"] is True
    assert game_row[0] != game_row[1]
    assert warning_event["warnings"] == []
    assert record["moves_count"] >= 9
    assert not (tmp_path / "logs").exists()
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.jsonl"))
    assert not list(tmp_path.glob("*.csv"))


def test_named_agents_use_default_database_without_output_files(tmp_path, monkeypatch):
    db_path = tmp_path / "data" / "tournament.sqlite"
    monkeypatch.setattr(arena, "TOURNAMENT_DB", db_path)
    assert main(["--a", "random", "--b", "heuristic", "--games", "2",
                 "--budget", "sims:1"]) == 0
    with sqlite3.connect(db_path) as connection:
        run_id, payload = connection.execute(
            "SELECT experiment_id, payload_json FROM games ORDER BY local_game_id LIMIT 1"
        ).fetchone()
        total, agents, matches, games = connection.execute(
            "SELECT (SELECT COUNT(*) FROM experiments WHERE experiment_id=?), "
            "(SELECT COUNT(*) FROM agents), (SELECT COUNT(*) FROM matches WHERE experiment_id=?), "
            "(SELECT COUNT(*) FROM games WHERE experiment_id=?)",
            (run_id, run_id, run_id),
        ).fetchone()
        summary = json.loads(connection.execute(
            "SELECT payload_json FROM events WHERE experiment_id=? AND event='arena_summary'", (run_id,)
        ).fetchone()[0])["summary"]
    assert total == matches == 1
    assert agents == 2
    assert games == 2
    assert summary["agent_a"] == "random"
    assert summary["agent_b"] == "heuristic"
    assert summary["as_black"]["games"] == summary["as_white"]["games"] == 1
    assert json.loads(payload)["moves"]
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "arena").exists()
    assert not list(tmp_path.glob("*.json"))
    assert not list(tmp_path.glob("*.jsonl"))
    assert not list(tmp_path.glob("*.csv"))
