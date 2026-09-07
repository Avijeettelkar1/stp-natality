# Architecture

## End-to-end system

```mermaid
flowchart LR
  Raw[DVC_RawFiles] --> Ingest[SchemaValidation]
  Ingest --> Prepare[CleanAndEngineer]
  Prepare --> Split[TemporalSplits]
  Split --> Train[CandidateTraining]
  Train --> Track[MLflow]
  Train --> Artifact[ModelArtifact]
  Artifact --> API[FastAPI]
  API --> User[ResearchUser]
  Split --> Drift[BatchDriftReport]
  Drift --> Review[HumanReview]
  Review -->|retrain| Train
```

## CI path

```mermaid
flowchart LR
  Push[GitPush] --> Format[FormatAndLint]
  Format --> Tests[UnitIntegrationTests]
  Tests --> Image[DockerBuild]
  Image --> Review[MergeReview]
```

The raw data, processed splits, model artifacts, and experiment database are not
committed to Git. Git contains the contracts and executable code needed to
recreate them.
