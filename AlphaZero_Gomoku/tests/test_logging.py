import pytest

from az.evaluator import summarise
from az.game_log import GameRecord, MoveRecord


def test_game_record_validation():
    record = GameRecord(
        game_id=1,
        timestamp="2026-10-28T00:00:00Z",
        rule_set="normal",
        first_player="A",
        second_player="B",
        winner="black",
        first_player_won=True,
        end_reason="five_in_row",
        moves_count=1,
        duration_sec=0.2,
    )
    record.validate()

    with pytest.raises(ValueError, match="Invalid end_reason"):
        record.end_reason = "invalid"
        record.validate()

    with pytest.raises(ValueError, match="Invalid winner"):
        record.end_reason = "five_in_row"
        record.winner = "unknown"
        record.validate()


def test_in_memory_game_records_keep_results_and_move_telemetry():
    move = MoveRecord(
        move_index=0, player=1, row=5, col=5, value_before=0.0,
        policy_top_k=[], simulations=12, latency_sec=0.125,
        nodes_visited=31, simulations_run=12,
    )
    records = [
        GameRecord(1, "2026-10-28T00:00:00Z", "normal", "A", "B", "black",
                   True, "five_in_row", 1, 0.2, moves=[move]),
        GameRecord(2, "2026-10-28T00:01:00Z", "normal", "B", "A", "draw",
                   False, "board_full", 2, 0.5),
    ]
    for record in records:
        record.validate()

    summary = summarise(records, "A", "B")
    assert (summary["wins"], summary["draws"], summary["losses"]) == (1, 1, 0)
    assert summary["score"] == 0.75
    assert summary["by_end_reason"]["five_in_row"] == 1
    assert summary["by_end_reason"]["board_full"] == 1
    assert summary["mean_game_duration_sec"] == 0.35
    assert summary["mean_moves_per_game"] == 1.5
    assert summary["telemetry"]["A"]["mean_latency_sec"] == 0.125
    assert summary["telemetry"]["A"]["nodes_visited"] == 31
    assert summary["telemetry"]["A"]["simulations_run"] == 12
