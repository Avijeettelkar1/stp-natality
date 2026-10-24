# Project Results

## Verification status

Verified on a bounded sample of 100,000 rows from each of the five real era
files. The run processed 499,259 valid records. These results do not claim to
represent a full 27-million-row training run.

## Data splits

| Split | Years | Rows |
|---|---|---:|
| Train | 1986–2009 | 299,431 |
| Validation | 2010–2015 | 75,083 |
| Test | 2016–2019 | 124,745 |

## Model comparison

| Model | Validation MAE | Validation R² | LBW sensitivity |
|---|---:|---:|---:|
| Mean baseline | 436.6 g | 0.000 | 0.0% |
| Ridge | 379.2 g | 0.310 | 28.7% |
| LightGBM | **366.6 g** | **0.351** | **34.3%** |

Selected model: LightGBM, with a 16.0% validation-MAE improvement over the mean
baseline.

## Held-out test results

| Metric | Result |
|---|---:|
| MAE | 362.8 g |
| RMSE | 469.7 g |
| R² | 0.389 |
| LBW sensitivity | 38.6% |
| LBW specificity | 99.0% |
| LBW precision | 76.4% |
| LBW F1 | 0.513 |

Term-LBW sensitivity was 1.3%, which rules out clinical screening use. Exact
machine-readable evidence is in `model_metrics.json`, with experiment lineage
in MLflow.
