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

## Setup (uv)

Install [uv](https://docs.astral.sh/uv/) if needed, then from the repo root:

```powershell
uv sync --extra dev
```

That creates `.venv` and installs the project plus test and notebook tools.

### Data and model (DVC on DagsHub)

The processed splits and the trained model are tracked with DVC on a private
DagsHub remote (`dvc` itself is not a core dependency — install it via the
`data` extra). To restore them you need **access to the DagsHub repo and a
token**:

```powershell
uv sync --extra data                        # installs dvc[http]
dvc remote modify dagshub --local auth basic
dvc remote modify dagshub --local user  <your-dagshub-username>
dvc remote modify dagshub --local password <your-dagshub-token>   # kept in .dvc/config.local (gitignored)
dvc pull                                     # restores data/processed/ and models/
```

Get the token from DagsHub → **Settings → Tokens** (you must be a collaborator
on the data repo). Without this access a fresh clone can still run the API, the
tests, and the serving container against a synthetic model
(`uv run python scripts/make_test_model.py models`), just not retrain on the
real data.

With the DVC-tracked data restored, run:

```powershell
uv run python scripts/run_pipeline.py --nrows 100000
uv run python scripts/train.py --sample-size 0
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Remove the row limit for the final data pipeline. Use `--sample-size 0` only
when the machine can train on the full split.

## Serve and monitor

```powershell
uv run uvicorn src.api.main:app --reload --port 8000
uv run python scripts/monitor_drift.py
```

For an Evidently drift report (training reference vs recent production inputs),
install the `monitoring` extra and run the monitoring job:

```powershell
uv sync --extra monitoring
$env:DATABASE_URL = "..."; uv run python scripts/run_drift_monitoring.py
# or offline, without a database:
uv run python scripts/run_drift_monitoring.py --current-parquet data/processed/test.parquet
```

It writes `reports/drift/latest.html` and `reports/drift/latest.json`.

Or start the API container:

```powershell
docker compose up --build
```

Swagger UI is available at `http://localhost:8000/docs`.

## Quality checks

```powershell
uv sync --extra ci
uv run pytest tests/ --cov=src
uv run black --check src scripts tests
uv run isort --check-only src scripts tests
uv run flake8 src scripts tests --max-line-length=88
```

See [`docs/final_report.md`](docs/final_report.md) and
[`docs/architecture.md`](docs/architecture.md) for the scientific narrative and
system design.
