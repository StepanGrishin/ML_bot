from datetime import datetime

from predictor.state import LeagueState


def test_snapshot_before_match():
    state = LeagueState()

    snapshot = state.snapshot(
        "a",
        "b",
        "Hard",
    )

    assert snapshot["elo_a"] == 1500
    assert snapshot["elo_b"] == 1500
    assert snapshot["h2h"] == 0
    assert snapshot["common_count"] == 0


def test_h2h_after_match():
    state = LeagueState()

    state.record_result(
        winner="a",
        loser="b",
        surface="Hard",
        date=datetime(2024, 1, 1),
    )

    assert state.h2h_score("a", "b") == 1.0
    assert state.h2h_score("b", "a") == -1.0
