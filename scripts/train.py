"""Train every candidate model and select the best one on validation MAE.

All candidates are logged to MLflow. The selected model is written to the
output directory together with its metadata, and the per-candidate results are
written to ``reports/model_metrics.json``.

Usage:
    python scripts/train.py
    python scripts/train.py --output-dir artifacts/challenger
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import load_splits  # noqa: E402
from src.models.candidates import (  # noqa: E402
    BASELINE_NAME,
    build_candidates,
    select_best,
)
from src.models.evaluate import evaluate_split, evaluate_subgroups  # noqa: E402
from src.models.pipeline import MODEL_FEATURES, get_X_y  # noqa: E402
from src.models.train import (  # noqa: E402
    EXPERIMENT_NAME,
    resolve_tracking_uri,
    train_and_log,
)

logger = logging.getLogger("train")

DATA_POINTER = PROJECT_ROOT / "data" / "processed.dvc"


def _sample(X, y, size: int | None, seed: int = 42):
    if size is None or len(X) <= size:
        return X, y
    rng = np.random.default_rng(seed)
    positions = rng.choice(len(X), size=size, replace=False)
    return X.iloc[positions], y.iloc[positions]


def read_data_version(pointer: Path = DATA_POINTER) -> str | None:
    """Return the DVC hash of the processed data, if the pointer exists."""
    if not pointer.exists():
        return None
    match = re.search(r"md5:\s*([0-9a-f]+)", pointer.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def main(
    sample_size: int | None,
    processed_dir: Path,
    output_dir: Path,
    report_path: Path,
) -> None:
    tracking_uri = resolve_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    train_df, val_df, test_df = load_splits(processed_dir)
    X_train, y_train = get_X_y(train_df)
    X_val, y_val = get_X_y(val_df)
    X_test, y_test = get_X_y(test_df)
    X_fit, y_fit = _sample(X_train, y_train, sample_size)

    data_version = read_data_version()
    tags = {"stage": "candidate", "data_contract": "natality-26-v1"}
    if data_version:
        tags["data_version"] = data_version
    if os.getenv("GITHUB_SHA"):
        tags["git_commit"] = os.environ["GITHUB_SHA"]

    candidates = build_candidates()
    results = {}
    for name, pipeline in candidates.items():
        results[name] = train_and_log(
            pipeline, name, X_fit, y_fit, X_val, y_val, tags=tags
        )

    best_name = select_best(results)
    best_pipeline = candidates[best_name]
    test_metrics = evaluate_split(best_pipeline, X_test, y_test, split_name="test")
    subgroup_metrics = evaluate_subgroups(best_pipeline, X_test, y_test)

    baseline_mae = results[BASELINE_NAME]["val_mae"]
    best_mae = results[best_name]["val_mae"]
    improvement_pct = 100 * (baseline_mae - best_mae) / baseline_mae

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "birth_weight_pipeline.joblib"
    metadata_path = output_dir / "model_metadata.json"

    joblib.dump(best_pipeline, model_path)
    now = datetime.now(timezone.utc)
    metadata = {
        "model_name": best_name,
        "model_version": now.strftime("%Y%m%d%H%M%S"),
        "trained_at_utc": now.isoformat(),
        "features": MODEL_FEATURES,
        "training_rows": len(X_fit),
        "data_contract": "natality-26-v1",
        "data_version": data_version,
        "git_commit": os.getenv("GITHUB_SHA"),
        "mlflow_run_id": results[best_name]["mlflow_run_id"],
        "val_mae": round(best_mae, 3),
        "validation_improvement_over_mean_pct": round(improvement_pct, 2),
        "subgroups": subgroup_metrics,
        **test_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_path.write_text(
        json.dumps({"candidates": results, "selected": metadata}, indent=2),
        encoding="utf-8",
    )
    for name, result in sorted(results.items(), key=lambda item: item[1]["val_mae"]):
        logger.info("  %-14s val_mae=%.1f g", name, result["val_mae"])
    logger.info("Selected %s; model written to %s", best_name, model_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Maximum training rows; 0 (default) uses all training rows.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "model_metrics.json",
    )
    args = parser.parse_args()
    main(
        sample_size=None if args.sample_size == 0 else args.sample_size,
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        report_path=args.report,
    )
