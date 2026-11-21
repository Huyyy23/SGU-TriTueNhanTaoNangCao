import pytest

from az.elo import paired_davidson_ci, recalculate_elo
from az.logging_v2 import (
    connect,
    initialize,
    record_tournament_agent,
    record_tournament_experiment,
    record_tournament_game,
    record_tournament_match,
)


def _experiment(connection):
    record_tournament_experiment(
        connection, experiment_id="exp", started_at="start", backend="python",
        rule_set="normal", seed=1, config={},
    )
    for agent_id in ("a", "b"):
        record_tournament_agent(
            connection, agent_id=agent_id, label=agent_id, algorithm="test",
            checkpoint_sha256=None, backend="python", rule_set="normal", config={},
        )
    record_tournament_match(
        connection, match_id="exp:pair", experiment_id="exp", agent_a_id="a", agent_b_id="b",
        pair_id="a|b:0", rule_set="normal", seed=None,
    )
    record_tournament_game(
        connection, experiment_id="exp", local_game_id=0, match_id="exp:pair",
        black_agent_id="a", white_agent_id="b",
        record={"game_id": 0, "rule_set": "normal", "winner": "black",
                "end_reason": "five_in_row", "moves_count": 1, "duration_sec": 0.1,
                "moves": [0]},
    )


def test_elo_recalculates_from_games_and_is_idempotent(tmp_path):
    database = tmp_path / "tournament.sqlite"
    initialize(database, "tournament")
    with connect(database) as connection:
        _experiment(connection)
        result = recalculate_elo(connection, "exp")
        assert result["ratings"] == {"a": 1216.0, "b": 1184.0}
        assert recalculate_elo(connection, "exp") == result
        assert connection.execute("SELECT count(*) FROM rating_events").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM rating_systems").fetchone()[0] == 1


def test_elo_rejects_invalid_k_factor(tmp_path):
    database = tmp_path / "tournament.sqlite"
    initialize(database, "tournament")
    with connect(database) as connection:
        _experiment(connection)
        with pytest.raises(ValueError, match="k_factor"):
            recalculate_elo(connection, "exp", k_factor=0)


def test_paired_davidson_uses_color_swapped_opening_clusters(tmp_path):
    database = tmp_path / "tournament.sqlite"
    initialize(database, "tournament")
    with connect(database) as connection:
        record_tournament_experiment(
            connection, experiment_id="paired", started_at="start", backend="python",
            rule_set="normal", seed=3, config={},
        )
        for agent_id, label in (("candidate-id", "candidate"), ("opponent-id", "opponent")):
            record_tournament_agent(
                connection, agent_id=agent_id, label=label, algorithm="test",
                checkpoint_sha256=None, backend="python", rule_set="normal", config={},
            )
        for pair_index in range(20):
            pair_id = f"candidate|opponent:{pair_index}"
            match_id = f"paired:{pair_index}"
            record_tournament_match(
                connection, match_id=match_id, experiment_id="paired",
                agent_a_id="candidate-id", agent_b_id="opponent-id", pair_id=pair_id,
                rule_set="normal", seed=pair_index, metadata={"games": 2},
            )
            for color_swap in range(2):
                candidate_black = color_swap == 0
                black_id = "candidate-id" if candidate_black else "opponent-id"
                white_id = "opponent-id" if candidate_black else "candidate-id"
                candidate_wins = pair_index < 15
                winner = ("black" if candidate_black else "white") if candidate_wins else (
                    "white" if candidate_black else "black"
                )
                record_tournament_game(
                    connection, experiment_id="paired", local_game_id=pair_index * 2 + color_swap,
                    match_id=match_id, black_agent_id=black_id, white_agent_id=white_id,
                    record={
                        "game_id": pair_index * 2 + color_swap, "rule_set": "normal",
                        "winner": winner, "end_reason": "five_in_row", "moves_count": 1,
                        "duration_sec": 0.1, "moves": [],
                    },
                )

        result = paired_davidson_ci(connection, ["paired"], "candidate", "opponent", seed=9)
        assert result["games"] == 40
        assert result["opening_pairs"] == 20
        assert result["wins"] == 30 and result["losses"] == 10 and result["draws"] == 0
        assert result["ci95"][0] > 0
        assert paired_davidson_ci(connection, ["paired"], "candidate", "opponent", seed=9) == result
