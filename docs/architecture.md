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

## Container view

One Docker container holds the frontend, the API and the model. See
[ADR 001](decisions/001-single-container-deployment.md).

```mermaid
flowchart LR
  User["User<br/>(web browser)"]
  subgraph Host["Hosting platform: Render web service (proposed)"]
    subgraph Container["Docker container: stp-natality-api"]
      UI["Static frontend<br/>index.html + JS<br/>served at /"]
      API["FastAPI app<br/>/predict, /health, /model/info, /docs"]
      Val["Input validation<br/>Pydantic PredictionRequest"]
      Feat["Feature preparation<br/>prepare_features()"]
      Model[("Model loaded in memory<br/>birth_weight_pipeline.joblib<br/>+ model_metadata.json")]
    end
  end
  User -- "HTTPS GET /" --> UI
  UI -- "fetch POST /predict (JSON)" --> API
  API --> Val --> Feat --> Model
```

## Prediction request sequence

```mermaid
sequenceDiagram
  actor U as User
  participant B as Browser (index.html)
  participant A as FastAPI /predict
  participant V as Pydantic schema
  participant S as ModelService
  participant M as Model pipeline (in memory)

  U->>B: Fill in form, click Predict
  B->>A: POST /predict (JSON)
  A->>V: Validate PredictionRequest
  alt invalid input
    V-->>A: Validation error
    A-->>B: 422 Unprocessable Entity
    B-->>U: Show field errors
  else valid input
    A->>S: predict(request)
    alt model not loaded
      S-->>A: RuntimeError
      A-->>B: 503 Service Unavailable
      B-->>U: Show "model unavailable"
    else model loaded
      S->>S: prepare_features() (parity, month sin/cos, flags)
      S->>M: predict(features)
      M-->>S: Weight in grams
      S-->>A: Rounded prediction
      A->>A: Assign risk_category (low / normal / macrosomia)
      A-->>B: 200 JSON (weight, risk_category, model_version, note)
      B-->>U: Show result + research-use note
    end
  end
```
