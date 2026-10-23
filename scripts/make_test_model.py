"""Create a small test model so the serving image can be built and smoke-tested.

The model is fitted on synthetic data and only reacts to gestation, sex and
plurality. It is used by CI and for local UI checks, never for real estimates.

Usage:
    uv run python scripts/make_test_model.py models
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.pipeline import MODEL_FEATURES, build_pipeline  # noqa: E402


def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame(0.0, index=range(n), columns=MODEL_FEATURES)
    X["gestation_weeks"] = rng.integers(28, 43, n)
    X["is_male"] = rng.integers(0, 2, n)
    X["plurality"] = rng.choice([1, 1, 1, 1, 2], n)
    y = (
        3350
        + 180 * (X["gestation_weeks"] - 39)
        + 120 * X["is_male"]
        - 900 * (X["plurality"] - 1)
        + rng.normal(0, 250, n)
    )

    pipeline = build_pipeline(LinearRegression())
    pipeline.fit(X, y)
    joblib.dump(pipeline, output_dir / "birth_weight_pipeline.joblib")
    metadata = {"model_name": "test-model", "model_version": "synthetic"}
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Test model written to {output_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "models"))
