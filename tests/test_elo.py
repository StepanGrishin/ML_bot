from predictor.elo import EloEngine


def test_initial_probability():
    elo = EloEngine()
    p = elo.probability("A", "B")
    assert abs(p - 0.5) < 1e-9


def test_winner_rating_increases():
    elo = EloEngine(k=32)
    before = elo.rating("A")
    elo.update("A", "B", "Hard")
    assert elo.rating("A") > before


def test_surface_rating_is_separate():
    elo = EloEngine(k=32)
    elo.update("A", "B", "Clay")
    assert elo.surface_rating("A", "Clay") > 1500
    assert elo.surface_rating("A", "Grass") == 1500
