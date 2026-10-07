from pathlib import Path
import pickle

from config import DATA_DIR
from predictor.data import load_matches
from predictor.state import LeagueState
from predictor.features import make_features


def build_dataset():
    matches = load_matches(DATA_DIR)
    state = LeagueState()

    rows = []

    for i, match in enumerate(matches):
        snapshot = state.snapshot(
            match.winner,
            match.loser,
            match.surface,
        )

        features = make_features(snapshot)

        rows.append({
            **features,
            "date": match.date,
            "surface": match.surface,
            "winner": match.winner,
            "loser": match.loser,
            "target": 1,
        })

        # Симметричная строка: для пары loser/winner
        # target = 0. Это делает модель менее зависимой
        # от того, кто записан первым.
        reverse_snapshot = state.snapshot(
            match.loser,
            match.winner,
            match.surface,
        )
        reverse_features = make_features(
            reverse_snapshot
        )

        rows.append({
            **reverse_features,
            "date": match.date,
            "surface": match.surface,
            "winner": match.winner,
            "loser": match.loser,
            "target": 0,
        })

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

        if (i + 1) % 5000 == 0:
            print(
                f"[DATASET] processed {i + 1}/{len(matches)}"
            )

    return rows, state


def save_dataset(rows, path):
    with path.open("wb") as f:
        pickle.dump(rows, f)
