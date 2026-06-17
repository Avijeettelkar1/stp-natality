"""Model training with MLflow experiment tracking for STP Natality.

Provides:
- train_and_log: Train a single pipeline, log params/metrics/model to MLflow.
- run_hpo: Optuna-based hyperparameter optimisation with per-trial MLflow runs.
"""

import logging
import time
from typing import Callable, Optional

import optuna
import pandas as pd

from src.models.evaluate import evaluate_split

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

EXPERIMENT_NAME = "stp-natality-phase3"


def train_and_log(
    pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: Optional[dict] = None,
    tags: Optional[dict] = None,
) -> dict:
    """Train a pipeline and log all params, metrics, and model to MLflow.

    Args:
        pipeline: Untrained sklearn Pipeline (preprocessor + model).
        model_name: Run name (e.g. "xgboost_baseline").
        X_train: Training feature matrix.
        y_train: Training target (weight_grams).
        X_val: Validation feature matrix.
        y_val: Validation target.
        params: Hyperparameters dict to log (optional).
        tags: Additional MLflow tags (optional).

    Returns:
        Dict containing all train_* and val_* metrics plus mlflow_run_id.
    """
    import mlflow

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("model_type", model_name)
        if tags:
            for k, v in tags.items():
                mlflow.set_tag(k, v)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_features", X_train.shape[1])
        if params:
            mlflow.log_params(params)

        logger.info("Training [%s] on %d rows...", model_name, len(X_train))
        t0 = time.time()
        pipeline.fit(X_train, y_train)
        elapsed = time.time() - t0
        mlflow.log_metric("train_time_sec", round(elapsed, 2))
        logger.info("  Done in %.1fs", elapsed)

        train_metrics = evaluate_split(
            pipeline, X_train, y_train, split_name="train", log_to_mlflow=True
        )
        val_metrics = evaluate_split(
            pipeline, X_val, y_val, split_name="val", log_to_mlflow=True
        )

        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = run.info.run_id

    return {**train_metrics, **val_metrics, "mlflow_run_id": run_id}


def run_hpo(
    model_name: str,
    create_pipeline_fn: Callable,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 30,
    direction: str = "minimize",
) -> tuple:
    """Run Optuna hyperparameter optimisation, logging each trial to MLflow.

    Each trial trains a pipeline returned by create_pipeline_fn(trial) and
    logs its val_mae as both the Optuna objective and an MLflow metric.

    Args:
        model_name: Prefix for MLflow run names (e.g. "xgboost_hpo").
        create_pipeline_fn: Callable(trial: optuna.Trial) → fitted pipeline.
            Must build and return an *untrained* sklearn Pipeline.
        X_train: Training feature matrix.
        y_train: Training target.
        X_val: Validation feature matrix.
        y_val: Validation target.
        n_trials: Number of Optuna trials.
        direction: Optuna direction ("minimize" for MAE).

    Returns:
        Tuple (best_params, study) where best_params is the best trial's
        params dict and study is the completed optuna.Study object.
    """
    import mlflow

    from src.models.evaluate import compute_regression_metrics

    mlflow.set_experiment(EXPERIMENT_NAME)

    def objective(trial: optuna.Trial) -> float:
        pipeline = create_pipeline_fn(trial)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        metrics = compute_regression_metrics(y_val, y_pred)
        mae = metrics["mae"]

        with mlflow.start_run(
            run_name=f"{model_name}_trial_{trial.number}", nested=True
        ):
            mlflow.set_tag("model_type", model_name)
            mlflow.set_tag("hpo_trial", str(trial.number))
            mlflow.log_params(trial.params)
            mlflow.log_metric("val_mae", mae)
            mlflow.log_metric("val_rmse", metrics["rmse"])
            mlflow.log_metric("val_r2", metrics["r2"])

        return mae

    with mlflow.start_run(run_name=f"{model_name}_hpo_{n_trials}trials"):
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("hpo", "optuna")
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best = study.best_trial
        mlflow.log_params(best.params)
        mlflow.log_metric("best_val_mae", best.value)
        logger.info(
            "HPO complete | best val_mae=%.1fg | params=%s",
            best.value,
            best.params,
        )

    return best.params, study
