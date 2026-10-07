from pathlib import Path
import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config import (
    MODEL_DIR,
    TEST_START_YEAR,
    VALIDATION_START_YEAR,
)
from .dataset import build_dataset, save_dataset
from predictor.features import FEATURE_COLUMNS, dataframe


def split_rows(rows):
    train = []
    validation = []
    test = []

    for row in rows:
        year = row["date"].year

        if year < VALIDATION_START_YEAR:
            train.append(row)
        elif year < TEST_START_YEAR:
            validation.append(row)
        else:
            test.append(row)

    return train, validation, test


def fit_model(rows):
    x = dataframe(rows)
    y = [row["target"] for row in rows]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )

    model.fit(x, y)
    return model


def evaluate(model, rows):
    x = dataframe(rows)
    y = [row["target"] for row in rows]
    p = model.predict_proba(x)[:, 1]
    pred = (p >= 0.5).astype(int)

    return {
        "rows": len(rows),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "log_loss": float(
            log_loss(y, p, labels=[0, 1])
        ),
        "brier": float(
            brier_score_loss(y, p)
        ),
        "auc": float(
            roc_auc_score(y, p)
        ) if len(set(y)) > 1 else None,
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows, state = build_dataset()
    save_dataset(
        rows,
        MODEL_DIR / "dataset.pkl",
    )

    train, validation, test = split_rows(rows)

    if not train:
        raise RuntimeError(
            "Training set is empty. "
            "Check historical data and dates."
        )

    print(
        f"train={len(train)} "
        f"validation={len(validation)} "
        f"test={len(test)}"
    )

    model = fit_model(train)

    metrics = {
        "validation": evaluate(
            model, validation
        ) if validation else None,
        "test": evaluate(
            model, test
        ) if test else None,
        "features": FEATURE_COLUMNS,
        "validation_start_year": VALIDATION_START_YEAR,
        "test_start_year": TEST_START_YEAR,
    }

    # Финальный production model обучаем на train+validation,
    # а test оставляем нетронутым для честной оценки.
    final_rows = train + validation
    final_model = fit_model(final_rows)

    joblib.dump(
        final_model,
        MODEL_DIR / "xgboost.pkl",
    )

    state.save(
        MODEL_DIR / "state.json"
    )

    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "model": "xgboost",
                "features": FEATURE_COLUMNS,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(json.dumps(
        metrics,
        indent=2,
        ensure_ascii=False,
    ))
    print("[OK] Model saved to models/xgboost.pkl")
    print("[OK] State saved to models/state.json")


if __name__ == "__main__":
    main()
