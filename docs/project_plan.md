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

## Team composition

The process model groups Data Science work into five role categories. Each team
member has a main role and supports others; every member reviews code and
contributes encountered challenges to the report.

| Role category (process model) | Responsibilities in this project | Main | Support |
|---|---|---|---|
| Strategy & Project Management | Scope, planning, advisor communication, checkpoint decisions, final report | `<Anjan>` | `<Avijeet>` |
| Data Management & Analysis | Data contract, ingestion, cleaning, feature engineering, temporal split, data quality | `<Avijeet>` | `<Anjan>` |
| Analytical Modeling | Candidate models, experiments in MLflow, evaluation, subgroup analysis | `<Dhruva>` | `<Rakshith>` |
| Data Infrastructure & Operations | CI/CD, container, deployment, DVC remote, monitoring, rollback | `<Dhruva>` | `<Rakshith>` |
| Domain Expertise | Clinical meaning of results, limitations, intended use | Advisors and literature | All members |


## Milestones

| Milestone | Stage | Status |
|---|---|---|
| Scope, requirements and success criteria defined | Business Understanding | Done |
| Data pipeline, quality checks and temporal split | Data Collection, Exploration, Preparation | Done |
| Candidate models trained and selected by validation MAE | Analysis | Done |
| Conditional-Go for the prototype recorded | Evaluation | Done; advisor confirmation open |
| Website, container, CI/CD and deployment to Render | Deployment | Done |
| Model and data versioned with DVC on DagsHub | Deployment | Model done; processed data open |
| Training in the pipeline with automatic model promotion | Deployment | Done|
| Scheduled monitoring and retraining trigger | Utilization | Open |
| Final report, presentation and demo | Project finalization | Open (`<04-11-2026>`) |

## Risks and responses

- Missing data artifact: stop numerical claims until files are restored.
- Memory pressure: use bounded development samples, Parquet, and explicit final
  run settings.
- Temporal schema changes: fail at ingestion and document missingness by era.
- Weak predictive performance: report the data ceiling; do not query the test
  set repeatedly.
- Clinical overclaim: show the research-use warning and prediction-time
  limitations in the API and documentation.
