"""
Colab-optimised training script — GPU-accelerated, all outputs saved to Google Drive.
Run from notebooks/05_colab_training.ipynb, not directly.
"""

import logging
import subprocess
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path("/content/stp-natality")
DRIVE_ROOT = Path("/content/drive/MyDrive/stp-natality")

DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
(DRIVE_ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
(DRIVE_ROOT / "mlruns").mkdir(exist_ok=True)

DB_URI      = f"sqlite:///{DRIVE_ROOT}/mlflow.db"
FIGURES_DIR = DRIVE_ROOT / "reports" / "figures"

sys.path.insert(0, str(ROOT))

# ── Logging — stdout + Drive log file ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(DRIVE_ROOT / "training.log"), mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── GPU detection ─────────────────────────────────────────────────────────────
try:
    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    HAS_GPU = True
    logger.info("GPU: %s", gpu_info)
except Exception:
    HAS_GPU = False
    logger.warning("No GPU detected — using CPU")

# LightGBM pip install does not include GPU support — use all CPU cores instead.
# XGBoost pip install includes CUDA — use GPU when available.
XGB_DEVICE = "cuda" if HAS_GPU else "cpu"
logger.info("LightGBM: CPU (n_jobs=-1) | XGBoost: %s", XGB_DEVICE)

# ── Imports ───────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from src.models.evaluate import evaluate_split
from src.models.pipeline import MODEL_FEATURES, build_pipeline, get_X_y
from src.models.train import EXPERIMENT_NAME, run_hpo, train_and_log

# ── MLflow — tracking DB and artifacts both on Drive ─────────────────────────
mlflow.set_tracking_uri(DB_URI)
client = mlflow.tracking.MlflowClient()
if client.get_experiment_by_name(EXPERIMENT_NAME) is None:
    client.create_experiment(
        EXPERIMENT_NAME,
        artifact_location=f"file:///{DRIVE_ROOT}/mlruns",
    )
mlflow.set_experiment(EXPERIMENT_NAME)
logger.info("MLflow DB   → %s", DB_URI)
logger.info("Artifacts   → %s/mlruns", DRIVE_ROOT)

# ── Load data (memory-efficient — only needed columns, float32) ───────────────
# Loading all 36 cols then extracting 21 peaks at ~12 GB.
# Loading 21 cols directly in float32 peaks at ~2.5 GB — works on any tier.
import gc

PROCESSED_DIR = ROOT / "data" / "processed"
COLS_NEEDED   = MODEL_FEATURES + ["weight_grams"]

def _load_split(path):
    df = pd.read_parquet(path, columns=COLS_NEEDED, engine="pyarrow")
    X  = df[MODEL_FEATURES].astype("float32")
    y  = df["weight_grams"].astype("float32")
    n  = len(df)
    del df
    gc.collect()
    return X, y, n

logger.info("Loading splits (float32, 21 cols only) ...")
X_train, y_train, n_train = _load_split(PROCESSED_DIR / "train.parquet")
logger.info("Train: %d rows x %d features", n_train, X_train.shape[1])
X_val,   y_val,   n_val   = _load_split(PROCESSED_DIR / "val.parquet")
logger.info("Val:   %d rows x %d features", n_val,   X_val.shape[1])
X_test,  y_test,  n_test  = _load_split(PROCESSED_DIR / "test.parquet")
logger.info("Test:  %d rows x %d features", n_test,  X_test.shape[1])
logger.info("LBW prevalence (train): %.2f%%", (y_train < 2500).mean() * 100)

# ── Baseline ──────────────────────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("BASELINE — DummyRegressor(mean)")
baseline = train_and_log(
    pipeline=build_pipeline(DummyRegressor(strategy="mean")),
    model_name="baseline_mean",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    tags={"phase": "3-colab", "stage": "baseline"},
)
logger.info("Baseline val_mae=%.1fg  val_r2=%.4f", baseline["val_mae"], baseline["val_r2"])

# ── Screening — 500K subsample ────────────────────────────────────────────────
logger.info("=" * 60)
logger.info("SCREENING — 500K subsample")
rng = np.random.default_rng(42)
idx = rng.choice(len(X_train), size=min(500_000, len(X_train)), replace=False)
X_screen, y_screen = X_train.iloc[idx], y_train.iloc[idx]

for name, model, scale in [
    ("ridge_default", Ridge(alpha=1.0), True),
    ("lgbm_default",  LGBMRegressor(n_estimators=300, random_state=42, verbose=-1, n_jobs=-1), False),
    ("xgb_default",   XGBRegressor(n_estimators=300, random_state=42, verbosity=0, device=XGB_DEVICE, tree_method="hist"), False),
]:
    m = train_and_log(
        pipeline=build_pipeline(model, scale_numeric=scale),
        model_name=name,
        X_train=X_screen, y_train=y_screen,
        X_val=X_val, y_val=y_val,
        tags={"phase": "3-colab", "stage": "screening"},
    )
    logger.info("%s  val_mae=%.1fg  val_r2=%.4f  lbw_sens=%.3f",
                name, m["val_mae"], m["val_r2"], m["val_sensitivity"])

# ── LightGBM HPO — 30 trials on full training set ────────────────────────────
logger.info("=" * 60)
logger.info("HPO — LightGBM  (30 trials, full 14M training set)")

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
            random_state=42, verbose=-1, n_jobs=-1,
        ),
        scale_numeric=False,
    )

lgbm_best_params, _ = run_hpo(
    model_name="lgbm_hpo",
    create_pipeline_fn=lgbm_pipeline_fn,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    n_trials=30,
)
logger.info("LightGBM best params: %s", lgbm_best_params)

# ── XGBoost HPO — 30 trials on full training set ─────────────────────────────
logger.info("=" * 60)
logger.info("HPO — XGBoost  (30 trials, full 14M training set)")

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
            random_state=42, verbosity=0, tree_method="hist", device=XGB_DEVICE,
        ),
        scale_numeric=False,
    )

xgb_best_params, _ = run_hpo(
    model_name="xgb_hpo",
    create_pipeline_fn=xgb_pipeline_fn,
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    n_trials=30,
)
logger.info("XGBoost best params: %s", xgb_best_params)

# ── Final models — best params, full training set ─────────────────────────────
logger.info("=" * 60)
logger.info("FINAL TRAINING — best params, full dataset")

lgbm_final_pipe = build_pipeline(
    LGBMRegressor(**lgbm_best_params, random_state=42, verbose=-1, n_jobs=-1),
    scale_numeric=False,
)
lgbm_final = train_and_log(
    pipeline=lgbm_final_pipe, model_name="lgbm_final",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    params=lgbm_best_params,
    tags={"phase": "3-colab", "stage": "final", "algo": "lightgbm"},
)
logger.info("LightGBM final  val_mae=%.1fg  val_r2=%.4f  lbw_sens=%.3f",
            lgbm_final["val_mae"], lgbm_final["val_r2"], lgbm_final["val_sensitivity"])

xgb_final_pipe = build_pipeline(
    XGBRegressor(**xgb_best_params, random_state=42, verbosity=0,
                 tree_method="hist", device=XGB_DEVICE),
    scale_numeric=False,
)
xgb_final = train_and_log(
    pipeline=xgb_final_pipe, model_name="xgb_final",
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    params=xgb_best_params,
    tags={"phase": "3-colab", "stage": "final", "algo": "xgboost"},
)
logger.info("XGBoost final   val_mae=%.1fg  val_r2=%.4f  lbw_sens=%.3f",
            xgb_final["val_mae"], xgb_final["val_r2"], xgb_final["val_sensitivity"])

# ── Pick best model ───────────────────────────────────────────────────────────
_candidates = {
    "lgbm_final": (lgbm_final_pipe, lgbm_final),
    "xgb_final":  (xgb_final_pipe,  xgb_final),
}
best_name, (best_pipe, best_val) = min(
    _candidates.items(), key=lambda kv: kv[1][1]["val_mae"]
)
logger.info("Best model: %s  (val_mae=%.1fg)", best_name, best_val["val_mae"])

# ── Test set evaluation (run once, on held-out test set) ─────────────────────
logger.info("=" * 60)
logger.info("TEST SET EVALUATION")
test_metrics = evaluate_split(best_pipe, X_test, y_test, split_name="test")

logger.info("  MAE            : %.1fg  %s  (target < 300g)",
            test_metrics["test_mae"], "PASS" if test_metrics["test_mae"] < 300 else "FAIL")
logger.info("  RMSE           : %.1fg", test_metrics["test_rmse"])
logger.info("  R2             : %.4f  %s  (target > 0.70)",
            test_metrics["test_r2"], "PASS" if test_metrics["test_r2"] > 0.70 else "FAIL")
logger.info("  LBW Sensitivity: %.3f  %s  (target > 0.85)",
            test_metrics["test_sensitivity"],
            "PASS" if test_metrics["test_sensitivity"] > 0.85 else "FAIL")
logger.info("  LBW Specificity: %.3f", test_metrics["test_specificity"])
logger.info("  LBW Precision  : %.3f", test_metrics["test_precision"])
logger.info("  LBW F1         : %.3f", test_metrics["test_f1"])

# ── Permutation importance — 50K subsample (avoids OOM on 4.6M val set) ──────
logger.info("=" * 60)
logger.info("FEATURE IMPORTANCE — permutation (50K subsample)")
from sklearn.inspection import permutation_importance

sample_idx = np.random.default_rng(42).choice(len(X_val), size=50_000, replace=False)
perm = permutation_importance(
    best_pipe, X_val.iloc[sample_idx], y_val.iloc[sample_idx],
    n_repeats=10, random_state=42,
    scoring="neg_mean_absolute_error", n_jobs=-1,
)
perm_df = pd.DataFrame(
    {"feature": X_val.columns, "importance": perm.importances_mean}
).sort_values("importance", ascending=False)
logger.info("\nTop 10 features:\n%s", perm_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(perm_df["feature"][:15][::-1], perm_df["importance"][:15][::-1])
ax.set_xlabel("Mean decrease in MAE (permutation importance)")
ax.set_title(f"Top 15 Features — {best_name}")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "feature_importance_permutation.png", dpi=150)
logger.info("Saved → %s", FIGURES_DIR / "feature_importance_permutation.png")

# ── SHAP — 5K subsample ───────────────────────────────────────────────────────
try:
    import shap
    logger.info("SHAP values on 5K subsample ...")
    shap_sample = X_val.sample(5000, random_state=42)
    X_shap = pd.DataFrame(
        best_pipe.named_steps["preprocessor"].transform(shap_sample),
        columns=MODEL_FEATURES,
    )
    explainer = shap.TreeExplainer(best_pipe.named_steps["model"])
    shap_values = explainer.shap_values(X_shap)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=150)
    logger.info("Saved → %s", FIGURES_DIR / "shap_summary.png")
except Exception as e:
    logger.warning("SHAP skipped: %s", e)

# ── Log test metrics and register model ───────────────────────────────────────
logger.info("=" * 60)
logger.info("REGISTERING MODEL")
run_id = best_val["mlflow_run_id"]
with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics(test_metrics)
    mlflow.set_tag("test_evaluated", "true")
    mlflow.set_tag("best_model", "true")

model_uri = f"runs:/{run_id}/model"
registered = mlflow.register_model(model_uri=model_uri, name="stp-natality-birth-weight")
logger.info("Registered: stp-natality-birth-weight v%s", registered.version)

# ── Smoke test ────────────────────────────────────────────────────────────────
preds = mlflow.sklearn.load_model(model_uri).predict(X_test.head())
logger.info("Smoke test (first 5 predictions):")
for i, (p, t) in enumerate(zip(preds, y_test.values[:5])):
    logger.info("  row %d: pred=%.0fg  actual=%.0fg  error=%.0fg", i, p, t, abs(p - t))

logger.info("=" * 60)
logger.info("DONE. All results saved to Google Drive: %s", DRIVE_ROOT)
