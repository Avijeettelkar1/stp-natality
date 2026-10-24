# 001 - Single-container deployment, plain HTML frontend

**Status:** Accepted
**Date:** 2026-10-23

## Context
We need a user-facing way to enter mother/pregnancy data and get a
birth-weight prediction, deployed through our CI/CD pipeline.

## Decision
- **Source of truth for the model:** the trained model files
  (`birth_weight_pipeline.joblib`, `model_metadata.json`) are versioned
  with DVC. `models.dvc` in Git points to the exact version; the files
  themselves are stored in the DagsHub DVC remote.
- **Delivery format:** during CD, the pipeline runs `dvc pull models.dvc`
  and copies the pulled files into the Docker image. At runtime the
  FastAPI backend loads the model from the image into memory. The image
  does not download anything when it starts.
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
- Each image is tagged with the commit SHA it was built from. That commit's
  `models.dvc` names the exact model version, so every deployment can be
  traced back to the model in DagsHub.
- Each image tag = exactly one model version, so rollback = redeploy the
  previous image.
- Frontend and API always deploy together, so they can't get out of sync.
- If traffic grew, we could split the model into its own service later.
