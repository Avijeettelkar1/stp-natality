# STP Natality

End-to-end university Data Science project for newborn-weight prediction,
structured around the MLOps-based Data Science process model.

## Current status

The data, training, serving, CI, and batch-monitoring paths are implemented.
A bounded real-data run processed 499,259 records and selected LightGBM
(test MAE 362.8 g, R² 0.389). Raw data and generated artifacts remain excluded
from Git.

The software is a research prototype, not a medical device or diagnosis.

## Lifecycle and artifacts

| Stage | Main artifacts |
|---|---|
| Business Understanding | `docs/business_understanding.md` |
| Data Collection, Exploration, Preparation | `src/data/`, notebooks 02–03, `docs/data_report.md` |
| Analysis | `src/models/`, `scripts/train.py`, notebook 04 |
| Evaluation | `docs/evaluation.md`, `reports/model_metrics.json` after training |
| Deployment | `src/api/`, `deployment/Dockerfile`, `docker-compose.yml` |
| Utilization | `src/monitoring/`, `scripts/monitor_drift.py` |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

With the five DVC-tracked data files restored, run:

```bash
python scripts/run_pipeline.py --nrows 100000
python scripts/train.py --sample-size 0
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Remove the row limit for the final data pipeline. Use `--sample-size 0` only
when the machine can train on the full split.

## Serve and monitor

```bash
uvicorn src.api.main:app --reload
python scripts/monitor_drift.py
```

Or start the API container:

```bash
docker compose up --build
```

Swagger UI is available at `http://localhost:8000/docs`.

## Quality checks

```bash
pip install -r requirements-ci.txt
pytest tests/ --cov=src
black --check src scripts tests
isort --check-only src scripts tests
flake8 src scripts tests --max-line-length=88
```

See [`docs/final_report.md`](docs/final_report.md) and
[`docs/architecture.md`](docs/architecture.md) for the scientific narrative and
system design.
