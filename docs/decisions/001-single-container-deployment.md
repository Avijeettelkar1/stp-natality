# 001 - Single-container deployment, plain HTML frontend

**Status:** Accepted
**Date:** 2026-10-23

## Context
We need a user-facing way to enter mother/pregnancy data and get a
birth-weight prediction, deployed through our CI/CD pipeline.

## Decision
- The trained model file (`birth_weight_pipeline.joblib`) is baked into
  the Docker image and loaded into memory by the FastAPI backend.
- The frontend is a single `index.html` (HTML, CSS, vanilla JS) served
  by FastAPI at `/`. It calls `POST /predict`.
- Result: one container, one deployment, one URL.

## Alternatives considered
- **Streamlit:** runs as its own server, so we would deploy and connect
  two services. More moving parts, more failure points, no benefit for
  a single form.
- **React:** adds a build step, node_modules and more Docker config for
  one form. Unnecessary complexity.
- **Separate model server (MLflow serving, BentoML, cloud endpoint):**
  useful when many apps share a model, it needs GPUs, or it must scale
  separately. For one small LightGBM model with low traffic it adds
  latency and a second service with no benefit.

## Consequences
- Each image tag = exactly one model version, so rollback = redeploy the
  previous image.
- Frontend and API always deploy together, so they can't get out of sync.
- If traffic grew, we could split the model into its own service later.
