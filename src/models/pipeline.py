"""Sklearn pipeline builder for the STP Natality birth weight predictor.

Defines the feature groups used for modeling and builds the preprocessing
ColumnTransformer and full sklearn Pipeline used in training and inference.

Feature selection rationale:
- Numeric features: imputed with column median (median is robust to outliers).
- Binary/ordinal features: imputed with most frequent value.
- Geographic columns (state, mother_*_state): excluded — high cardinality;
  target encoding deferred to a future iteration.
- Substance use columns: excluded — >99% missing in the provided files.
- record_weight: excluded — sampling metadata, not a clinical predictor.
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Feature Groups ─────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    "gestation_weeks",  # strongest predictor (r≈+0.38)
    "mother_age",
    "weight_gain_pounds",  # maternal nutrition proxy
    "born_alive_alive",
    "born_alive_dead",
    "born_dead",
    "ever_born",
    "father_age",  # ~25% missing → imputed
    "plurality",  # singleton=1, twins=2, …
    "parity",  # derived: total prior births
    "year_fix",  # captures secular trend
    "month",
    "birth_month_sin",
    "birth_month_cos",
]

BINARY_FEATURES = [
    "is_male",
    "mother_married",
    "lmp_known",
    "is_multiple_birth",
    "gestation_preterm",
    "gestation_post_term",
    "mother_age_group",
]

MODEL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES

# Excluded feature groups (documented for traceability)
GEO_FEATURES_EXCLUDED = ["state", "mother_residence_state", "mother_birth_state"]
SUBSTANCE_FEATURES_EXCLUDED = [
    "cigarette_use",
    "cigarettes_per_day",
    "alcohol_use",
    "drinks_per_week",
]
META_EXCLUDED = ["record_weight", "source_year", "source_file", "lmp"]


def build_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    """Build the feature preprocessing ColumnTransformer.

    Args:
        scale_numeric: Apply StandardScaler after median imputation.
            Use True for linear models; False for tree-based models.

    Returns:
        Unfitted ColumnTransformer ready for use in a Pipeline.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "binary",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]),
                BINARY_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(model, scale_numeric: bool = False) -> Pipeline:
    """Combine the preprocessor with a regressor into a full sklearn Pipeline.

    Args:
        model: Any sklearn-compatible regressor.
        scale_numeric: Passed to build_preprocessor (use True for linear models).

    Returns:
        Untrained sklearn Pipeline: preprocessor → model.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def get_X_y(
    df: pd.DataFrame,
    target_col: str = "weight_grams",
) -> tuple:
    """Extract the model feature matrix X and target y from a split DataFrame.

    Selects only MODEL_FEATURES; casts nullable integer columns to float so
    that all sklearn transformers receive a uniform numeric dtype.

    Args:
        df: Train, val, or test split loaded from Parquet.
        target_col: Name of the target column (default: weight_grams).

    Returns:
        Tuple (X, y) where X is a float64 DataFrame of MODEL_FEATURES.

    Raises:
        ValueError: If target_col is not present in df.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

    available = [f for f in MODEL_FEATURES if f in df.columns]
    missing = [f for f in MODEL_FEATURES if f not in df.columns]
    if missing:
        logger.warning("Missing expected features (will be skipped): %s", missing)

    # Cast nullable Int types to float64 for sklearn compatibility
    X = df[available].astype("float64")
    y = df[target_col]

    logger.info(
        "get_X_y: X=%s | y=%s | features=%d",
        X.shape,
        y.shape,
        len(available),
    )
    return X, y
