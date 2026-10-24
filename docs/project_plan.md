# Project Plan

## Objective

Deliver a reproducible newborn-weight Data Science prototype across all stages
of the provided MLOps-based process model. Scientific documentation is produced
with each technical artifact rather than after implementation.

## Work packages

| Package | Output | Completion condition |
|---|---|---|
| Business | Scope and criteria | Team/advisor approval |
| Data | Valid temporal splits | Pipeline and tests pass on restored raw data |
| Analysis | Selected model | MLflow runs and held-out metrics recorded |
| Evaluation | Go/No-Go record | Business and subgroup evidence reviewed |
| Deployment | Containerized API | Integration tests and smoke test pass |
| Utilization | Drift and maintenance demo | Report generated and runbook exercised |
| Reporting | Final scientific report | Evidence, figures, references, reflection reviewed |

## Roles

- Project coordination/domain translation: scope, evidence, advisor decisions.
- Data engineering: data contract, pipeline, quality, and lineage.
- Analytical modeling: experiments, evaluation, interpretation.
- Infrastructure/operations: CI, API, container, monitoring, maintenance.

All members review code and contribute encountered challenges to the report.

## Immediate sequence

1. Restore the five DVC-tracked raw files.
2. Run a bounded development pipeline and all tests.
3. Run the final data preparation and model comparison.
4. Review results and record the evaluation decision.
5. Package the approved model, smoke-test the API, and generate drift evidence.
6. Insert verified results into the final report and complete peer/advisor review.

## Risks and responses

- Missing data artifact: stop numerical claims until files are restored.
- Memory pressure: use bounded development samples, Parquet, and explicit final
  run settings.
- Temporal schema changes: fail at ingestion and document missingness by era.
- Weak predictive performance: report the data ceiling; do not query the test
  set repeatedly.
- Clinical overclaim: show the research-use warning and prediction-time
  limitations in the API and documentation.
