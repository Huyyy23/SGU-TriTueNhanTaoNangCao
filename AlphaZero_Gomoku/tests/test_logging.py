import json
from pathlib import Path
import pytest

from az.game_log import GameLogger, GameRecord, MoveRecord


def test_game_record_validation():
    # Valid record
    rec = GameRecord(
        game_id=1,
        timestamp="2026-10-28T00:00:00Z",
        rule_set="normal",
        black_player="AZ_new",
        white_player="AZ_best",
        winner="black",
        first_player_won=True,
        end_reason="five_in_row",
        num_moves=25,
        duration_sec=3.5,
    )
    rec.validate()

    # Invalid end_reason
    with pytest.raises(ValueError, match="Invalid end_reason"):
        invalid_rec = GameRecord(
            game_id=2,
            timestamp="2026-10-28T00:00:00Z",
            rule_set="normal",
            black_player="AZ_new",
            white_player="AZ_best",
            winner="black",
            first_player_won=True,
            end_reason="invalid_reason",
            num_moves=10,
            duration_sec=1.0,
        )
        invalid_rec.validate()

    # Invalid winner
    with pytest.raises(ValueError, match="Invalid winner"):
        invalid_rec2 = GameRecord(
            game_id=3,
            timestamp="2026-10-28T00:00:00Z",
            rule_set="normal",
            black_player="AZ_new",
            white_player="AZ_best",
            winner="someone_else",
            first_player_won=True,
            end_reason="five_in_row",
            num_moves=10,
            duration_sec=1.0,
        )
        invalid_rec2.validate()


def test_game_logger_write_and_summary(tmp_path: Path):
    log_file = tmp_path / "games.jsonl"
    summary_file = tmp_path / "summary.json"
    logger = GameLogger(log_file, log_moves=False)

    # Game 1: AZ_new (Black) beats AZ_best (White)
    rec1 = GameRecord(
        game_id=1,
        timestamp="2026-10-28T00:01:00Z",
        rule_set="normal",
        black_player="AZ_new",
        white_player="AZ_best",
        winner="black",
        first_player_won=True,
        end_reason="five_in_row",
        num_moves=30,
        duration_sec=4.0,
        nn_calls=6000,
        mean_batch=16.0,
    )
    # Game 2: AZ_best (Black) beats AZ_new (White)
    rec2 = GameRecord(
        game_id=2,
        timestamp="2026-10-28T00:02:00Z",
        rule_set="normal",
        black_player="AZ_best",
        white_player="AZ_new",
        winner="black",
        first_player_won=True,
        end_reason="five_in_row",
        num_moves=26,
        duration_sec=3.0,
        nn_calls=5200,
        mean_batch=16.0,
    )
    # Game 3: AZ_new (White) beats AZ_best (Black)
    rec3 = GameRecord(
        game_id=3,
        timestamp="2026-10-28T00:03:00Z",
        rule_set="normal",
        black_player="AZ_best",
        white_player="AZ_new",
        winner="white",
        first_player_won=False,
        end_reason="board_full",
        num_moves=121,
        duration_sec=15.0,
        nn_calls=24200,
        mean_batch=15.5,
    )

    logger.log_game(rec1)
    logger.log_game(rec2)
    logger.log_game(rec3)

    records = logger.read_records(log_file)
    assert len(records) == 3
    assert records[0]["game_id"] == 1
    assert records[0]["winner"] == "black"
    assert "moves" not in records[0]

    # Test summary calculation for AZ_new vs AZ_best
    summary = logger.save_summary(summary_file, agent_a="AZ_new", agent_b="AZ_best")
    assert summary["total_games"] == 3
    assert summary["agent_a"] == "AZ_new"
    # AZ_new won Game 1 (as black) and Game 3 (as white) -> 2/3 wins
    assert summary["overall_win_rate_a"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["as_black"]["games"] == 1
    assert summary["as_black"]["wins"] == 1
    assert summary["as_black"]["win_rate"] == 1.0

    assert summary["as_white"]["games"] == 2
    assert summary["as_white"]["wins"] == 1
    assert summary["as_white"]["win_rate"] == 0.5

    assert summary["by_end_reason"]["five_in_row"] == 2
    assert summary["by_end_reason"]["board_full"] == 1

    # Check that summary file exists and can be loaded
    assert summary_file.is_file()
    with open(summary_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["total_games"] == 3


def test_game_logger_with_moves(tmp_path: Path):
    log_file = tmp_path / "moves.jsonl"
    logger = GameLogger(log_file, log_moves=True)

    moves = [
        MoveRecord(
            move_index=0,
            player=1,
            row=5,
            col=5,
            value_before=0.1,
            policy_top_k=[{"move": 60, "prob": 0.4}],
            simulations=200,
        )
    ]
    rec = GameRecord(
        game_id=1,
        timestamp="2026-10-28T00:00:00Z",
        rule_set="normal",
        black_player="AZ",
        white_player="Random",
        winner="black",
        first_player_won=True,
        end_reason="five_in_row",
        num_moves=1,
        duration_sec=0.2,
        moves=moves,
    )
    logger.log_game(rec)

    records = logger.read_records(log_file)
    assert len(records) == 1
    assert "moves" in records[0]
    assert len(records[0]["moves"]) == 1
    assert records[0]["moves"][0]["row"] == 5
