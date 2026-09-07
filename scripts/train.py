"""Train and select a birth-weight model from processed temporal splits.

Usage:
    python scripts/train.py --sample-size 500000
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import load_splits  # noqa: E402
from src.models.evaluate import evaluate_split, evaluate_subgroups  # noqa: E402
from src.models.pipeline import (  # noqa: E402
    MODEL_FEATURES,
    build_pipeline,
    get_X_y,
)
from src.models.train import EXPERIMENT_NAME, train_and_log  # noqa: E402

logger = logging.getLogger("train")


def _sample(X, y, size: int | None, seed: int = 42):
    if size is None or len(X) <= size:
        return X, y
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(X), size=size, replace=False)
    return X.iloc[positions], y.iloc[positions]


def main(sample_size: int | None, processed_dir: Path, output_dir: Path) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    train_df, val_df, test_df = load_splits(processed_dir)
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)
    X_fit, y_fit = _sample(X_train, y_train, sample_size)

    candidates = {
        "dummy_mean": build_pipeline(DummyRegressor(strategy="mean")),
        "ridge": build_pipeline(Ridge(alpha=1.0), scale_numeric=True),
        "lightgbm": build_pipeline(
            LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
    }

    results = {}
    for name, pipeline in candidates.items():
        results[name] = train_and_log(
            pipeline,
            name,
            X_fit,
            y_fit,
            X_val,
            y_val,
            tags={"stage": "candidate", "data_contract": "natality-26-v1"},
        )

    best_name = min(results, key=lambda name: results[name]["val_mae"])
    best_pipeline = candidates[best_name]
    test_metrics = evaluate_split(best_pipeline, X_test, y_test, split_name="test")
    subgroup_metrics = evaluate_subgroups(best_pipeline, X_test, y_test)

    baseline_mae = results["dummy_mean"]["val_mae"]
    best_mae = results[best_name]["val_mae"]
    improvement_pct = 100 * (baseline_mae - best_mae) / baseline_mae

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "birth_weight_pipeline.joblib"
    metadata_path = output_dir / "model_metadata.json"
    results_path = PROJECT_ROOT / "reports" / "model_metrics.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, model_path)
    metadata = {
        "model_name": best_name,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": MODEL_FEATURES,
        "training_rows": len(X_fit),
        "data_contract": "natality-26-v1",
        "validation_improvement_over_mean_pct": round(improvement_pct, 2),
        "subgroups": subgroup_metrics,
        **test_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    results_path.write_text(
        json.dumps({"candidates": results, "selected": metadata}, indent=2),
        encoding="utf-8",
    )
    logger.info("Selected %s; model written to %s", best_name, model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500_000,
        help="Maximum training rows; use 0 for all rows.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    args = parser.parse_args()
    main(
        sample_size=None if args.sample_size == 0 else args.sample_size,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
    )
