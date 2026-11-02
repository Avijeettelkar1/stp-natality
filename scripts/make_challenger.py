"""Train a challenger model for shadow evaluation.

The champion is LightGBM; this trains the XGBoost candidate on the same
processed training split and saves it under ``models/challenger/`` with its own
metadata. Shipped in the serving image and loaded via CHALLENGER_MODEL_PATH, it
runs in shadow mode alongside the champion (never affecting the response).

Usage:
    uv run python scripts/make_challenger.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.models.candidates import build_candidates  # noqa: E402
from src.models.pipeline import MODEL_FEATURES, get_X_y  # noqa: E402

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
OUT_DIR = PROJECT_ROOT / "models" / "challenger"
CANDIDATE = "xgboost"


def main() -> None:
    df = pd.read_parquet(TRAIN_PATH)
    X, y = get_X_y(df)
    pipeline = build_candidates()[CANDIDATE]
    print(f"Training challenger '{CANDIDATE}' on {len(X):,} rows ...")
    pipeline.fit(X, y)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_DIR / "birth_weight_pipeline.joblib")
    metadata = {
        "model_name": CANDIDATE,
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(X)),
        "data_contract": "natality-26-v1",
        "role": "challenger",
        "features": MODEL_FEATURES,
    }
    (OUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Challenger saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
