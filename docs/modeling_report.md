# Modeling Report

## Status

The modeling workflow is implemented, but final numerical results are pending
restoration of the five raw `newborn_*` files. Earlier numbers in `reports/`
are not accepted as verified because their feature descriptions include fields
outside the provided 26-column schema.

## Reproducible experiment

1. Run `python scripts/run_pipeline.py --nrows 100000` for a development split.
2. Run `python scripts/train.py --sample-size 500000`.
3. Inspect runs with
   `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
4. Replace development limits with `--nrows` omitted and `--sample-size 0` only
   when sufficient memory and time are available.

The candidates are:

- `DummyRegressor(mean)`: minimum baseline.
- Ridge regression: interpretable linear comparison.
- LightGBM: non-linear tree ensemble for tabular data.

Selection uses validation MAE. The selected model is evaluated once on the
temporally held-out test set. The script writes:

- `models/birth_weight_pipeline.joblib`
- `models/model_metadata.json`
- `reports/model_metrics.json`
- MLflow parameters, metrics, and model artifacts

## Metrics

| Metric | Purpose |
|---|---|
| MAE | Primary, interpretable average error in grams |
| RMSE | Highlights large errors |
| R² | Fraction of observed variance explained |
| LBW sensitivity/specificity | Secondary threshold behavior |

Final values must be copied from `reports/model_metrics.json`; they must not be
entered from memory or from the superseded reports.

## Intended use and limitations

The model is a research decision-support prototype for late-pregnancy
assessment. It is not validated for clinical deployment. Major limitations are
the mismatch between completed gestation in training data and current gestation
at prediction time, missing clinical measurements, temporal collection changes,
and potential demographic proxy bias.
