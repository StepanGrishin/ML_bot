from datetime import date
from pathlib import Path
import json

import joblib

from config import DATA_DIR, MODEL_DIR
from .data import load_matches, normalize_name, display_name
from .state import LeagueState
from .features import make_features, dataframe
from .common_opponents import common_opponent_details


class TennisPredictor:
    def __init__(
        self,
        model_path: Path | None = None,
        state_path: Path | None = None,
    ):
        self.model_path = model_path or (
            MODEL_DIR / "xgboost.pkl"
        )
        self.state_path = state_path or (
            MODEL_DIR / "state.json"
        )

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. "
                f"Run `python training/train.py` first."
            )

        if not self.state_path.exists():
            raise FileNotFoundError(
                f"State not found: {self.state_path}. "
                f"Run `python training/train.py` first."
            )

        self.model = joblib.load(self.model_path)
        self._all_matches = None
        self._state_cache = {}

    def _matches(self):
        if self._all_matches is None:
            self._all_matches = load_matches(DATA_DIR)
        return self._all_matches

    def _build_state(self, before_date=None):
        key = str(before_date) if before_date else "all"

        if key in self._state_cache:
            return self._state_cache[key]

        state = LeagueState()

        for match in self._matches():
            if before_date is not None:
                if match.date.date() >= before_date:
                    break

            state.record_result(
                winner=match.winner,
                loser=match.loser,
                surface=match.surface,
                date=match.date,
                winner_rank=match.winner_rank,
                loser_rank=match.loser_rank,
                winner_rank_points=match.winner_rank_points,
                loser_rank_points=match.loser_rank_points,
            )

        self._state_cache[key] = state
        return state

    def predict(
        self,
        player_a,
        player_b,
        surface="Hard",
        before_date=None,
    ):
        if before_date is None:
            before_date = date.today()

        a = normalize_name(player_a)
        b = normalize_name(player_b)

        state = self._build_state(before_date)

        snapshot = state.snapshot(a, b, surface)
        row = make_features(snapshot)
        x = dataframe([row])

        probability_a = float(
            self.model.predict_proba(x)[0][1]
        )
        probability_b = 1.0 - probability_a

        winner = (
            player_a
            if probability_a >= probability_b
            else player_b
        )

        return {
            "player_a": player_a,
            "player_b": player_b,
            "probability_a": probability_a * 100.0,
            "probability_b": probability_b * 100.0,
            "winner": winner,
            "elo_a": snapshot["elo_a"],
            "elo_b": snapshot["elo_b"],
            "surface_elo_a": snapshot["surface_elo_a"],
            "surface_elo_b": snapshot["surface_elo_b"],
            "common_count": snapshot["common_count"],
            "h2h": snapshot["h2h"],
            "details": common_opponent_details(
                state, a, b
            ),
            "features": row,
        }
