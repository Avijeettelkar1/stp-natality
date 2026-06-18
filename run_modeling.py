"""
Standalone training script — runs the full Phase 3 modeling pipeline.
Equivalent to executing notebooks/04_modeling.ipynb end-to-end.

Usage:
    python run_modeling.py

Results are logged to mlflow.db (MLflow SQLite backend).
Run `mlflow ui --backend-store-uri sqlite:///mlflow.db` to inspect.
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("modeling_run.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import mlflow
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.data.split import load_splits
from src.models.evaluate import evaluate_split
from src.models.pipeline import MODEL_FEATURES, build_pipeline, get_X_y
from src.models.train import EXPERIMENT_NAME, run_hpo, train_and_log

# ── MLflow setup ───────────────────────────────────────────────────────────────
DB_URI = f"sqlite:///{ROOT}/mlflow.db"
mlflow.set_tracking_uri(DB_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
logger.info("MLflow experiment: %s  |  URI: %s", EXPERIMENT_NAME, DB_URI)

# ── Load data ─────────────────────────────────────────────────────────────────
logger.info("Loading splits from data/processed/ ...")
PROCESSED_DIR = ROOT / "data" / "processed"
train_df, val_df, test_df = load_splits(PROCESSED_DIR)
logger.info("Train=%d  Val=%d  Test=%d", len(train_df), len(val_df), len(test_df))

X_train, y_train = get_X_y(train_df)
X_val,   y_val   = get_X_y(val_df)
X_test,  y_test  = get_X_y(test_df)
logger.info("LBW prevalence (train): %.2f%%", (y_train < 2500).mean() * 100)

# ── Baseline ──────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("BASELINE — DummyRegressor(mean)")
baseline_metrics = train_and_log(
    pipeline=build_pipeline(DummyRegressor(strategy="mean")),
    model_name="baseline_mean",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    tags={"phase": "3", "stage": "baseline"},
)
logger.info("Baseline val_mae=%.1fg  val_r2=%.4f", baseline_metrics["val_mae"], baseline_metrics["val_r2"])

# ── Screening ─────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("SCREENING — 500K subsample")
SCREEN_N = 500_000
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), size=min(SCREEN_N, len(X_train)), replace=False)
X_screen, y_screen = X_train.iloc[idx], y_train.iloc[idx]

CANDIDATES = {
    "ridge_default": (Ridge(alpha=1.0), True),
    "lgbm_default":  (LGBMRegressor(n_estimators=300, random_state=42, verbose=-1), False),
    "xgb_default":   (XGBRegressor(n_estimators=300, random_state=42, verbosity=0), False),
}
screening_results = {}
for name, (model, scale) in CANDIDATES.items():
    m = train_and_log(
        pipeline=build_pipeline(model, scale_numeric=scale),
        model_name=name,
        X_train=X_screen, y_train=y_screen,
        X_val=X_val, y_val=y_val,
        tags={"phase": "3", "stage": "screening"},
    )
    screening_results[name] = m
    logger.info("%s  val_mae=%.1fg  val_r2=%.4f  lbw_sens=%.3f",
                name, m["val_mae"], m["val_r2"], m["val_sensitivity"])

screen_df = pd.DataFrame([
    {"model": k, "val_mae": v["val_mae"], "val_r2": v["val_r2"],
     "val_sensitivity": v["val_sensitivity"]}
    for k, v in screening_results.items()
]).sort_values("val_mae")
logger.info("\nScreening summary:\n%s", screen_df.to_string(index=False))

# ── LightGBM HPO ──────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("HPO — LightGBM  (30 trials, full training set)")

def lgbm_pipeline_fn(trial):
    return build_pipeline(
        LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            num_leaves=trial.suggest_int("num_leaves", 31, 255),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 20, 200),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            random_state=42, verbose=-1,
        ),
        scale_numeric=False,
    )

lgbm_best_params, lgbm_study = run_hpo(
    model_name="lgbm_hpo",
    create_pipeline_fn=lgbm_pipeline_fn,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    n_trials=30,
)
logger.info("LightGBM best params: %s", lgbm_best_params)

# ── XGBoost HPO ───────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("HPO — XGBoost  (30 trials, full training set)")

def xgb_pipeline_fn(trial):
    return build_pipeline(
        XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 200, 1000, step=100),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 50),
            random_state=42, verbosity=0, tree_method="hist",
        ),
        scale_numeric=False,
    )

xgb_best_params, xgb_study = run_hpo(
    model_name="xgb_hpo",
    create_pipeline_fn=xgb_pipeline_fn,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    n_trials=30,
)
logger.info("XGBoost best params: %s", xgb_best_params)

# ── Final models (full training set) ──────────────────────────────────────────
logger.info("=" * 60)
logger.info("FINAL TRAINING — best params, full dataset")

lgbm_final_pipe = build_pipeline(
    LGBMRegressor(**lgbm_best_params, random_state=42, verbose=-1), scale_numeric=False
)
lgbm_final = train_and_log(
    pipeline=lgbm_final_pipe, model_name="lgbm_final",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    params=lgbm_best_params,
    tags={"phase": "3", "stage": "final", "algo": "lightgbm"},
)
logger.info("LightGBM final  val_mae=%.1fg  val_r2=%.4f  lbw_sens=%.3f",
            lgbm_final["val_mae"], lgbm_final["val_r2"], lgbm_final["val_sensitivity"])

xgb_final_pipe = build_pipeline(
    XGBRegressor(**xgb_best_params, random_state=42, verbosity=0, tree_method="hist"),
    scale_numeric=False,
)
xgb_final = train_and_log(
    pipeline=xgb_final_pipe, model_name="xgb_final",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    params=xgb_best_params,
    tags={"phase": "3", "stage": "final", "algo": "xgboost"},
)
logger.info("XGBoost final   val_mae=%.1fg  val_r2=%.4f  lbw_sens=%.3f",
            xgb_final["val_mae"], xgb_final["val_r2"], xgb_final["val_sensitivity"])

# ── Pick best ─────────────────────────────────────────────────────────────────
candidates = {
    "lgbm_final": (lgbm_final_pipe, lgbm_final),
    "xgb_final":  (xgb_final_pipe,  xgb_final),
}
best_name = min(candidates, key=lambda k: candidates[k][1]["val_mae"])
best_pipe, best_val = candidates[best_name]
logger.info("Best model: %s  (val_mae=%.1fg)", best_name, best_val["val_mae"])

# ── Test evaluation (run ONCE) ─────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("TEST SET EVALUATION")
test_metrics = evaluate_split(best_pipe, X_test, y_test, split_name="test")

mae_ok  = test_metrics["test_mae"] < 300
r2_ok   = test_metrics["test_r2"] > 0.70
sens_ok = test_metrics["test_sensitivity"] > 0.85

logger.info("  MAE            : %.1fg  %s  (target < 300g)", test_metrics["test_mae"],  "PASS" if mae_ok  else "FAIL")
logger.info("  RMSE           : %.1fg",                       test_metrics["test_rmse"])
logger.info("  R2             : %.4f  %s  (target > 0.70)",  test_metrics["test_r2"],   "PASS" if r2_ok   else "FAIL")
logger.info("  LBW Sensitivity: %.3f  %s  (target > 0.85)",  test_metrics["test_sensitivity"], "PASS" if sens_ok else "FAIL")
logger.info("  LBW Specificity: %.3f",                        test_metrics["test_specificity"])
logger.info("  LBW Precision  : %.3f",                        test_metrics["test_precision"])
logger.info("  LBW F1         : %.3f",                        test_metrics["test_f1"])
logger.info("  LBW prevalence : %.2f%%",                      test_metrics["test_lbw_prevalence_pct"])

# ── Feature importance ────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("FEATURE IMPORTANCE")
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

perm = permutation_importance(
    best_pipe, X_val, y_val,
    n_repeats=10, random_state=42,
    scoring="neg_mean_absolute_error", n_jobs=-1,
)
perm_df = pd.DataFrame(
    {"feature": X_val.columns, "importance": perm.importances_mean}
).sort_values("importance", ascending=False)
logger.info("\n%s", perm_df.head(10).to_string(index=False))

figures_dir = ROOT / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(perm_df["feature"][:15][::-1], perm_df["importance"][:15][::-1])
ax.set_xlabel("Mean decrease in MAE (permutation importance)")
ax.set_title(f"Top 15 Features — {best_name}")
plt.tight_layout()
plt.savefig(figures_dir / "feature_importance_permutation.png", dpi=150)
logger.info("Saved permutation importance plot.")

try:
    import shap
    shap_sample = X_val.sample(5000, random_state=42)
    final_model = best_pipe.named_steps["model"]
    preprocessor = best_pipe.named_steps["preprocessor"]
    X_shap = pd.DataFrame(preprocessor.transform(shap_sample), columns=MODEL_FEATURES)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_shap)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "shap_summary.png", dpi=150)
    logger.info("Saved SHAP summary plot.")
except Exception as e:
    logger.warning("SHAP skipped: %s", e)

# ── Register model ────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("REGISTERING MODEL")
run_id = best_val["mlflow_run_id"]
with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics(test_metrics)
    mlflow.set_tag("test_evaluated", "true")
    mlflow.set_tag("best_model", "true")

MODEL_REGISTRY_NAME = "stp-natality-birth-weight"
model_uri = f"runs:/{run_id}/model"
registered = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)
logger.info("Registered: %s  v%s  (run_id=%s)", MODEL_REGISTRY_NAME, registered.version, run_id)

# ── Smoke test ────────────────────────────────────────────────────────────────
loaded = mlflow.sklearn.load_model(model_uri)
preds = loaded.predict(X_test.head())
logger.info("Smoke test predictions (first 5 rows):")
for i, (p, t) in enumerate(zip(preds, y_test.values[:5])):
    logger.info("  row %d: pred=%.0fg  actual=%.0fg  error=%.0fg", i, p, t, abs(p - t))

logger.info("=" * 60)
logger.info("DONE. Results in mlflow.db — run: mlflow ui --backend-store-uri sqlite:///mlflow.db")
