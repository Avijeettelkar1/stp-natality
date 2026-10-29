"""Candidate models compared in every training run.

Adding a model means adding one entry to ``build_candidates``. The training
script trains every candidate and keeps the one with the lowest validation
MAE (``select_best``). ``dummy_mean`` is the baseline every model must beat.
"""

from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.models.pipeline import build_pipeline

BASELINE_NAME = "dummy_mean"


def build_candidates(random_state: int = 42) -> dict[str, Pipeline]:
    """Return a fresh, unfitted pipeline for every candidate model."""
    return {
        BASELINE_NAME: build_pipeline(DummyRegressor(strategy="mean")),
        "ridge": build_pipeline(Ridge(alpha=1.0), scale_numeric=True),
        "random_forest": build_pipeline(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=random_state,
            )
        ),
        "lightgbm": build_pipeline(
            LGBMRegressor(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=63,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
        "xgboost": build_pipeline(
            XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
            )
        ),
    }


def select_best(results: dict[str, dict]) -> str:
    """Return the name of the candidate with the lowest validation MAE."""
    if not results:
        raise ValueError("No candidate results to select from.")
    return min(results, key=lambda name: results[name]["val_mae"])
