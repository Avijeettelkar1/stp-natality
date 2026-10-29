"""Run the promotion gate: compare the newly trained model with the live one.

Both models are evaluated on the current test split. The decision and its
reasons are written to a JSON report, to the GitHub Actions job summary, and
to the step outputs (``promote=true|false``) so the workflow can act on it.

When the challenger is promoted and MLflow points to a remote server, the
challenger's MLflow run is registered as a new version in the model registry.

Usage:
    python scripts/promote_model.py --champion-dir models \\
        --challenger-dir artifacts/challenger
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import load_splits  # noqa: E402
from src.models.promotion import PromotionRules, decide, scorecard  # noqa: E402

logger = logging.getLogger("promote")

MODEL_FILE = "birth_weight_pipeline.joblib"
METADATA_FILE = "model_metadata.json"
REGISTERED_MODEL_NAME = "birth-weight-model"


def load_model(directory: Path):
    """Return (pipeline, metadata) from a model directory, or (None, None)."""
    model_path = directory / MODEL_FILE
    if not model_path.exists():
        return None, None
    metadata_path = directory / METADATA_FILE
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.exists()
        else {}
    )
    return joblib.load(model_path), metadata


def register_challenger(run_id: str) -> str | None:
    """Register the challenger run in the MLflow model registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    if not os.getenv("MLFLOW_TRACKING_URI", "").startswith("http"):
        logger.info("No remote MLflow server configured; skipping registration.")
        return None
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL_NAME)
    try:
        MlflowClient().set_registered_model_alias(
            REGISTERED_MODEL_NAME, "champion", version.version
        )
    except Exception as error:  # alias support depends on the MLflow server
        logger.warning("Could not set the 'champion' alias: %s", error)
    return str(version.version)


def write_summary(decision: dict, challenger_meta: dict, champion_meta: dict) -> None:
    """Append a readable summary to the GitHub Actions job page."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    verdict = "PROMOTED" if decision["promote"] else "REJECTED"
    lines = [
        f"## Promotion gate: {verdict}",
        "",
        "| | Model | Version | Test MAE (g) | Test R² |",
        "|---|---|---|---|---|",
        f"| Challenger | {challenger_meta.get('model_name')} | "
        f"{challenger_meta.get('model_version')} | "
        f"{decision['challenger']['test_mae']:.1f} | "
        f"{decision['challenger']['test_r2']:.3f} |",
    ]
    if decision["champion"]:
        lines.append(
            f"| Live (champion) | {champion_meta.get('model_name')} | "
            f"{champion_meta.get('model_version')} | "
            f"{decision['champion']['test_mae']:.1f} | "
            f"{decision['champion']['test_r2']:.3f} |"
        )
    lines += ["", "**Reasons:**", ""] + [f"- {r}" for r in decision["reasons"]]
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_output(name: str, value: str) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def main(
    champion_dir: Path,
    challenger_dir: Path,
    processed_dir: Path,
    report_path: Path,
    register: bool,
) -> bool:
    _, _, test_df = load_splits(processed_dir)

    challenger_model, challenger_meta = load_model(challenger_dir)
    if challenger_model is None:
        raise FileNotFoundError(f"No challenger model in {challenger_dir}")
    champion_model, champion_meta = load_model(champion_dir)

    decision = decide(
        challenger=scorecard(challenger_model, test_df),
        champion=scorecard(champion_model, test_df) if champion_model else None,
        baseline_improvement_pct=challenger_meta[
            "validation_improvement_over_mean_pct"
        ],
        rules=PromotionRules(),
    ).to_dict()
    decision["challenger_model"] = challenger_meta.get("model_name")
    decision["champion_model"] = (champion_meta or {}).get("model_name")

    if decision["promote"] and register and challenger_meta.get("mlflow_run_id"):
        decision["registry_version"] = register_challenger(
            challenger_meta["mlflow_run_id"]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_summary(decision, challenger_meta, champion_meta or {})
    write_output("promote", "true" if decision["promote"] else "false")
    write_output("challenger", str(challenger_meta.get("model_name")))

    logger.info("Promotion decision: %s", "PROMOTE" if decision["promote"] else "KEEP")
    for reason in decision["reasons"]:
        logger.info("  - %s", reason)
    return decision["promote"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument(
        "--challenger-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "challenger",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "reports" / "promotion_decision.json",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register a promoted challenger in the MLflow model registry.",
    )
    args = parser.parse_args()
    main(
        champion_dir=args.champion_dir,
        challenger_dir=args.challenger_dir,
        processed_dir=args.processed_dir,
        report_path=args.report,
        register=args.register,
    )
