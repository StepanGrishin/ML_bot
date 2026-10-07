"""
Простой expanding-window backtest.

Для каждого test year модель обучается только на более ранних годах.
Это медленнее обычного train.py, но лучше отражает production-сценарий.
"""

from collections import defaultdict
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

from .dataset import build_dataset
from predictor.features import dataframe


def fit(rows):
    model = XGBClassifier(
        n_estimators=350,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    model.fit(
        dataframe(rows),
        [r["target"] for r in rows],
    )
    return model


def main():
    rows, _ = build_dataset()

    by_year = defaultdict(list)
    for row in rows:
        by_year[row["date"].year].append(row)

    years = sorted(by_year)

    for test_year in years[3:]:
        train = [
            row
            for year in years
            if year < test_year
            for row in by_year[year]
        ]
        test = by_year[test_year]

        if not train or not test:
            continue

        model = fit(train)

        p = model.predict_proba(
            dataframe(test)
        )[:, 1]
        y = [r["target"] for r in test]

        print(
            f"{test_year}: "
            f"accuracy={accuracy_score(y, p >= 0.5):.4f} "
            f"logloss={log_loss(y, p):.4f} "
            f"n={len(test)}"
        )


if __name__ == "__main__":
    main()
