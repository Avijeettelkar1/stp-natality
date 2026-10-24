# Modeling Report

## Status

The workflow was run on a bounded real-data sample of 100,000 rows from each of
the five era files (500,000 raw rows). After target validation, the temporal
splits contained 299,431 training, 75,083 validation, and 124,745 test rows.
This is the reproducible project result; it is not presented as a full
27-million-record experiment.

## Reproducible experiment

1. Run `python scripts/run_pipeline.py --nrows 100000`.
2. Run `python scripts/train.py --sample-size 0`.
3. Inspect runs with
   `mlflow ui --backend-store-uri sqlite:///mlflow.db`.
4. A full-data rerun is optional and requires a more memory-efficient ingestion
   strategy or a machine with more than the available 16 GB RAM.

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

## Results

| Model | Validation MAE | Validation R² | LBW sensitivity |
|---|---:|---:|---:|
| Mean baseline | 436.6 g | 0.000 | 0.0% |
| Ridge | 379.2 g | 0.310 | 28.7% |
| LightGBM | **366.6 g** | **0.351** | **34.3%** |

LightGBM improved validation MAE by 16.0% over the mean baseline and was selected.
On the held-out 2016–2019 test split it achieved MAE 362.8 g, RMSE 469.7 g,
R² 0.389, LBW sensitivity 38.6%, specificity 99.0%, precision 76.4%, and
F1 0.513.

Subgroup evaluation exposed the most important limitation. Term-birth LBW
sensitivity was only 1.3%, compared with 59.3% for preterm births. Female and
male MAE were 356.2 g and 369.1 g respectively. These results support prototype
deployment but not clinical use.

## Intended use and limitations

The model is a research decision-support prototype for late-pregnancy
assessment. It is not validated for clinical deployment. Major limitations are
the mismatch between completed gestation in training data and current gestation
at prediction time, missing clinical measurements, temporal collection changes,
and potential demographic proxy bias.
