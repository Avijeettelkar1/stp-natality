"""
Run the full Phase 2 data preparation pipeline:
  ingest → preprocess → feature_engineering → temporal_split → save

Produces data/processed/train.parquet, val.parquet, test.parquet.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --nrows 200000   # dev/test run (fast)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Project root on path ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ingest import load_all_files, get_data_summary
from src.data.preprocess import run_preprocessing
from src.data.feature_engineering import engineer_features
from src.data.split import temporal_split, save_splits, get_feature_columns

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def main(nrows_per_file: int | None = None) -> None:
    RAW_DATA_DIR = PROJECT_ROOT / "newborn_data"
    OUT_DIR = PROJECT_ROOT / "data" / "processed"

    logger.info("=" * 60)
    logger.info("STP Natality — Phase 2 Data Pipeline")
    logger.info("=" * 60)
    logger.info("Raw data dir : %s", RAW_DATA_DIR)
    logger.info("Output dir   : %s", OUT_DIR)
    logger.info("Rows per file: %s", nrows_per_file if nrows_per_file else "ALL")

    t_start = time.time()

    # ── Step 1: Ingestion ──────────────────────────────────────────────────────
    logger.info("\n--- Step 1: Ingestion ---")
    t0 = time.time()
    df_raw = load_all_files(data_dir=RAW_DATA_DIR, nrows_per_file=nrows_per_file)
    summary = get_data_summary(df_raw)
    logger.info(
        "Loaded %d rows × %d cols (%.1f MB) in %.1fs",
        summary["n_rows"],
        summary["n_cols"],
        summary["memory_mb"],
        time.time() - t0,
    )

    # ── Step 2: Preprocessing ──────────────────────────────────────────────────
    logger.info("\n--- Step 2: Preprocessing ---")
    t0 = time.time()
    df_clean = run_preprocessing(df_raw)
    logger.info(
        "Preprocessing done: %d → %d rows (%.2f%% removed) in %.1fs",
        summary["n_rows"],
        len(df_clean),
        100 * (summary["n_rows"] - len(df_clean)) / max(summary["n_rows"], 1),
        time.time() - t0,
    )

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    logger.info("\n--- Step 3: Feature Engineering ---")
    t0 = time.time()
    df_feat = engineer_features(df_clean)
    feature_cols = get_feature_columns(df_feat)
    logger.info(
        "Feature engineering done: %d feature columns in %.1fs",
        len(feature_cols),
        time.time() - t0,
    )

    # ── Step 4: Temporal Split ────────────────────────────────────────────────
    logger.info("\n--- Step 4: Temporal Split ---")
    t0 = time.time()
    train, val, test = temporal_split(df_feat)
    logger.info("Split done in %.1fs", time.time() - t0)

    # ── Step 5: Save Splits ───────────────────────────────────────────────────
    logger.info("\n--- Step 5: Saving Parquet Splits ---")
    t0 = time.time()
    save_splits(train, val, test, out_dir=OUT_DIR)
    logger.info("Saved in %.1fs", time.time() - t0)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs (%.1f min)", total_time, total_time / 60)
    logger.info("  train.parquet : %d rows", len(train))
    logger.info("  val.parquet   : %d rows", len(val))
    logger.info("  test.parquet  : %d rows", len(test))
    logger.info("  Features      : %d", len(feature_cols))
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run STP Natality data pipeline")
    parser.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Rows per file to load (None=full dataset)",
    )
    args = parser.parse_args()
    main(nrows_per_file=args.nrows)
