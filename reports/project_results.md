# Project Results

## Verification status

Final results are pending a clean rerun after the five DVC-tracked raw files are
restored. The prior report has been superseded because it described features
that do not exist in the provided 26-column schema.

Run:

```bash
python scripts/run_pipeline.py
python scripts/train.py --sample-size 0
python scripts/monitor_drift.py
```

Then source all numerical claims from:

- `reports/model_metrics.json`
- `reports/drift_report.json`
- the corresponding MLflow run

Do not copy earlier MAE, R², sensitivity, row-count, or feature-importance
values into the final report unless the clean run reproduces them.
