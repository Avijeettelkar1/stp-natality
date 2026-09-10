# Deployment Guide

## Prerequisite

Generate `models/birth_weight_pipeline.joblib` and
`models/model_metadata.json` with `uv run python scripts/train.py`. The model files are
intentionally excluded from Git.

## Local service

```powershell
uv run uvicorn src.api.main:app --reload --port 8000
```

Open `http://localhost:8000/docs`. `GET /health` reports whether the model is
loaded. `POST /predict` returns HTTP 503 when no model artifact exists.

## Container

```bash
docker compose up --build
```

Compose mounts the local read-only `models/` directory into the API container.

## Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"gestation_weeks":39,"mother_age":29,"weight_gain_pounds":27,
       "born_alive_alive":1,"ever_born":2,"father_age":31,"plurality":1,
       "year_fix":2021,"month":6,"is_male":true,"mother_married":true}'
```

## Rollback

Keep each approved model and metadata pair in versioned artifact storage. To
roll back, restore both files, restart the API, and verify `/model/info` and a
known smoke-test request. Never mix metadata from one model with another.
