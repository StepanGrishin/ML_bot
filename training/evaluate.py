import json
from pathlib import Path

from config import MODEL_DIR
from .train import evaluate, split_rows
from .dataset import build_dataset


def main():
    model_path = MODEL_DIR / "xgboost.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            "Model missing. Run: python -m training.train"
        )

    import joblib
    model = joblib.load(model_path)

    rows, _ = build_dataset()
    train, validation, test = split_rows(rows)

    result = {
        "validation": evaluate(model, validation),
        "test": evaluate(model, test),
    }

    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
