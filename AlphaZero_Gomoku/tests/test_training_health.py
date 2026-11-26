from az.training_health import outcome_counts, outcome_imbalance_message


def test_guard_is_disabled_by_default_rate():
    assert outcome_imbalance_message({"black": 500}, min_games=0, max_side_win_rate=1.01) is None


def test_guard_waits_for_min_games():
    assert outcome_imbalance_message(
        {"black": 9}, min_games=10, max_side_win_rate=0.98
    ) is None


def test_guard_reports_one_sided_outcomes():
    message = outcome_imbalance_message(
        {"black": 500, "white": 0, "draw": 0}, min_games=500, max_side_win_rate=0.98
    )
    assert message is not None
    assert "black won 500/500" in message
    assert "100.0%" in message


def test_guard_accepts_balanced_outcomes_and_draws():
    counts = {"black": 260, "white": 240, "draw": 500}
    assert outcome_imbalance_message(counts, min_games=500, max_side_win_rate=0.98) is None


def test_guard_trips_just_at_the_threshold():
    counts = {"black": 490, "white": 10}
    assert outcome_imbalance_message(counts, min_games=500, max_side_win_rate=0.98) is not None
    assert outcome_imbalance_message(
        {"black": 489, "white": 11}, min_games=500, max_side_win_rate=0.98
    ) is None


def test_outcome_counts_reads_winner_rows():
    class FakeCursor:
        def fetchall(self):
            return [("black", 3), ("white", 1), ("draw", 2)]

    class FakeConnection:
        def execute(self, sql, params):
            assert "FROM games WHERE run_id" in sql and params == ("run",)
            return FakeCursor()

    assert outcome_counts(FakeConnection(), "run") == {"black": 3, "white": 1, "draw": 2}
