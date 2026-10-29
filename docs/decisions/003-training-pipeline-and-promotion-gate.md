# 003 - Training in the pipeline with a promotion gate

**Status:** Accepted
**Date:** 2026-10-28

## Context
The process model deploys an ML production pipeline, not only a model: the
pipeline trains the model from the feature store, the model is versioned in a
model registry, and the serving component makes it available. Until now the
deployed model was trained on a laptop and uploaded by hand.

## Decision
- On every merge to `main` (and on manual or monitoring-triggered runs), the
  CD workflow trains all candidates in `src/models/candidates.py` on the
  processed data pulled from DVC. Every candidate run is logged to MLflow on
  DagsHub (model metadata store).
- The candidate with the lowest validation MAE becomes the challenger.
- A promotion gate (`src/models/promotion.py`) compares the challenger with the
  live model (champion) on the same test split. The challenger replaces the
  champion only if:
  1. it beats the mean-prediction baseline on validation by at least 10%,
  2. its test MAE is at least 1 g lower than the champion's, and
  3. no subgroup with at least 500 test rows gets more than 5% worse.
- A promoted model is pushed with DVC, `models.dvc` is committed to `main`, and
  the model is registered as a new version of `birth-weight-model` in the
  MLflow model registry. The deploy job then builds and deploys it.
- The decision and its reasons appear in the GitHub Actions job summary and in
  the `training-reports` artifact.

## Alternatives considered
- **Always deploy the newest model:** simple, but a worse model could replace a
  better one without anyone noticing.
- **Champion/Challenger (shadow) deployment:** runs both models in production
  and only monitors the challenger. It is the strongest validation but needs
  live traffic and ground truth, which arrives only after birth. It is a
  possible extension; the offline gate is the first step.
- **Manual approval of every model:** safest, but not automated.

## Consequences
- Adding a model means adding one entry to `build_candidates()`; the pipeline
  trains it and decides whether it is deployed.
- Every promotion is traceable: commit SHA, `models.dvc` hash, MLflow run and
  registry version.
- The test split serves as a fixed benchmark across runs, so it is no longer
  used only once. Selection inside a run still uses validation.
- To roll back a model, revert the promotion commit (restores the previous
  `models.dvc`) or redeploy an earlier image with the rollback workflow.
