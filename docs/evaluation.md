# Business Evaluation

## Decision rule

The project may proceed to prototype deployment when:

- the candidate model improves validation MAE over the mean baseline;
- the held-out test metrics and subgroup results are recorded;
- the API serves the exact training feature contract;
- known limitations are visible to users.

Failure to meet an aspirational clinical metric is a finding, not a reason to
hide or repeatedly query the test set.

## Evidence reviewed

- LightGBM validation MAE: 366.6 g versus 436.6 g for the mean baseline
  (16.0% improvement).
- Held-out test: MAE 362.8 g, RMSE 469.7 g, R² 0.389.
- LBW screening: sensitivity 38.6%, specificity 99.0%, precision 76.4%.
- Sex subgroup MAE: 356.2 g for female and 369.1 g for male records.
- LBW sensitivity: 59.3% preterm but only 1.3% term.
- Model version `20260907153537`, data contract `natality-26-v1`.

These values come from a bounded 499,259-record real-data run, not the full
dataset.

## Current decision

**Conditional Go for software demonstration; No-Go for clinical use.**

The model beats the agreed baseline and the end-to-end pipeline, deployment,
and monitoring are demonstrable. Clinical use is rejected because overall LBW
sensitivity is low, term-LBW sensitivity is effectively unusable, the dataset
lacks important clinical measurements, and gestational age has a
prediction-time mismatch.
