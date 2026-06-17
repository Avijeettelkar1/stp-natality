"""Evaluation metrics for the STP Natality birth weight predictor.

Provides standard regression metrics (MAE, RMSE, R²) and clinical metrics
(LBW sensitivity, specificity, precision, F1) derived from applying the
clinical LBW threshold (< 2500g) to continuous weight predictions.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

# Clinical weight thresholds (grams)
LBW_THRESHOLD = 2500.0
MACROSOMIA_THRESHOLD = 4000.0


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict:
    """Compute MAE, RMSE, and R² for continuous birth weight predictions.

    Args:
        y_true: Observed birth weights in grams.
        y_pred: Predicted birth weights in grams.

    Returns:
        Dict with keys: mae, rmse, r2 (all rounded).
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 4),
    }


def compute_clinical_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    threshold: float = LBW_THRESHOLD,
) -> dict:
    """Compute clinical LBW classification metrics from continuous predictions.

    Binarises both true and predicted values against the LBW threshold and
    computes sensitivity (recall for LBW), specificity, precision, and F1.

    Args:
        y_true: Observed birth weights in grams.
        y_pred: Predicted birth weights in grams.
        threshold: LBW classification boundary in grams (default: 2500g).

    Returns:
        Dict with keys: lbw_prevalence_pct, sensitivity, specificity,
        precision, f1, n_lbw_true, n_lbw_pred.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    true_lbw = (y_true_arr < threshold).astype(int)
    pred_lbw = (y_pred_arr < threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(true_lbw, pred_lbw, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )

    return {
        "lbw_prevalence_pct": round(float(true_lbw.mean() * 100), 2),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "precision": round(float(precision), 4),
        "f1": round(float(f1), 4),
        "n_lbw_true": int(true_lbw.sum()),
        "n_lbw_pred": int(pred_lbw.sum()),
    }


def evaluate_split(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "val",
    log_to_mlflow: bool = False,
) -> dict:
    """Evaluate a trained pipeline on one data split.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X: Feature matrix.
        y: Target series (weight_grams).
        split_name: Label prefix for metric keys, e.g. "train", "val", "test".
        log_to_mlflow: If True, log all metrics to the active MLflow run.

    Returns:
        Dict of all metrics prefixed with split_name (e.g. "val_mae").
    """
    y_pred = pipeline.predict(X)

    reg = compute_regression_metrics(y, y_pred)
    clin = compute_clinical_metrics(y, y_pred)
    combined = {f"{split_name}_{k}": v for k, v in {**reg, **clin}.items()}

    logger.info(
        "[%s] MAE=%.1fg | RMSE=%.1fg | R²=%.4f | LBW Sensitivity=%.1f%%",
        split_name.upper(),
        reg["mae"],
        reg["rmse"],
        reg["r2"],
        clin["sensitivity"] * 100,
    )

    if log_to_mlflow:
        try:
            import mlflow

            mlflow.log_metrics(combined)
        except Exception as exc:
            logger.warning("MLflow metric logging skipped: %s", exc)

    return combined
