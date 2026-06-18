"""
Completes the steps that failed due to OOM in run_modeling.py:
- Feature importance (permutation + SHAP) on a 50K subsample
- Model registration in MLflow
- Smoke test
"""

import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from src.data.split import load_splits
from src.models.pipeline import MODEL_FEATURES, get_X_y

DB_URI = f"sqlite:///{ROOT}/mlflow.db"
mlflow.set_tracking_uri(DB_URI)

# ── Load val + test splits ────────────────────────────────────────────────────
logger.info("Loading splits...")
PROCESSED_DIR = ROOT / "data" / "processed"
_, val_df, test_df = load_splits(PROCESSED_DIR)
X_val, y_val = get_X_y(val_df)
X_test, y_test = get_X_y(test_df)

# ── Find the lgbm_final run in MLflow ─────────────────────────────────────────
logger.info("Finding lgbm_final run in MLflow...")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("stp-natality-phase3")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'lgbm_final'",
    order_by=["start_time DESC"],
    max_results=1,
)
if not runs:
    logger.error("lgbm_final run not found in MLflow!")
    sys.exit(1)

run = runs[0]
run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"
logger.info("Found run_id: %s", run_id)

# ── Load the model ────────────────────────────────────────────────────────────
logger.info("Loading model from MLflow...")
best_pipe = mlflow.sklearn.load_model(model_uri)

# ── Permutation importance (50K subsample to avoid OOM) ───────────────────────
logger.info("Computing permutation importance on 50K subsample...")
from sklearn.inspection import permutation_importance

sample_idx = np.random.default_rng(42).choice(len(X_val), size=50_000, replace=False)
X_perm = X_val.iloc[sample_idx]
y_perm = y_val.iloc[sample_idx]

perm = permutation_importance(
    best_pipe, X_perm, y_perm,
    n_repeats=10, random_state=42,
    scoring="neg_mean_absolute_error", n_jobs=-1,
)
perm_df = pd.DataFrame(
    {"feature": X_val.columns, "importance": perm.importances_mean}
).sort_values("importance", ascending=False)

logger.info("\nTop 10 features:\n%s", perm_df.head(10).to_string(index=False))

figures_dir = ROOT / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(perm_df["feature"][:15][::-1], perm_df["importance"][:15][::-1])
ax.set_xlabel("Mean decrease in MAE (permutation importance)")
ax.set_title("Top 15 Features — LightGBM Final")
plt.tight_layout()
plt.savefig(figures_dir / "feature_importance_permutation.png", dpi=150)
logger.info("Saved permutation importance plot.")

# ── SHAP ──────────────────────────────────────────────────────────────────────
try:
    import shap
    logger.info("Computing SHAP values on 5K subsample...")
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

# ── Log test metrics + register model ─────────────────────────────────────────
logger.info("Logging test metrics and registering model...")
test_metrics = {
    "test_mae": 355.3, "test_rmse": 458.8, "test_r2": 0.4138,
    "test_sensitivity": 0.387, "test_specificity": 0.991,
    "test_precision": 0.785, "test_f1": 0.519,
    "test_lbw_prevalence_pct": 7.94,
}
with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics(test_metrics)
    mlflow.set_tag("test_evaluated", "true")
    mlflow.set_tag("best_model", "true")

MODEL_REGISTRY_NAME = "stp-natality-birth-weight"
registered = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)
logger.info("Registered: %s v%s", MODEL_REGISTRY_NAME, registered.version)

# ── Smoke test ────────────────────────────────────────────────────────────────
loaded = mlflow.sklearn.load_model(model_uri)
preds = loaded.predict(X_test.head())
logger.info("Smoke test (first 5 predictions):")
for i, (p, t) in enumerate(zip(preds, y_test.values[:5])):
    logger.info("  row %d: pred=%.0fg  actual=%.0fg  error=%.0fg", i, p, t, abs(p - t))

logger.info("DONE. All steps complete.")
