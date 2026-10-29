# Architecture

This document maps the system to the components of the MLOps-based Data Science
Process Model (MLOps-DSPM). Components drawn with a **dashed border** are
planned and not yet in production; everything else is implemented.

## 1. MLOps components (development and production environment)

```mermaid
flowchart LR
  classDef planned stroke-dasharray: 5 5,color:#666

  subgraph DEV["Development environment"]
    direction TB
    Repo["Source code repository<br/>GitHub"]
    ExpPipe["Experimental ML pipeline<br/>src/models, notebooks"]
    Meta[("Model metadata store<br/>MLflow on DagsHub")]
    Reg[("Model registry<br/>MLflow registry + models.dvc")]
  end

  subgraph CICD["CI/CD component: GitHub Actions"]
    direction TB
    CI["CI: lint, tests,<br/>container smoke test"]
    Train["ML production pipeline:<br/>train candidates + promotion gate"]
    Deploy["Build, push, deploy,<br/>live verification"]
  end

  subgraph PROD["Production environment"]
    direction TB
    Sources[("Data sources<br/>raw natality files (DVC)")]
    DataPipe["Data pipeline<br/>scripts/run_pipeline.py"]
    FStore[("Feature store<br/>processed splits (DVC on DagsHub)")]
    Serving["Serving component<br/>FastAPI on Render"]
    Shadow["Shadow challenger<br/>(champion/challenger)"]:::planned
    Log[("Prediction log + feedback<br/>Postgres")]:::planned
    Monitor["Monitoring component<br/>drift + shadow report, Grafana"]:::planned
    CT["CT component<br/>retraining trigger"]:::planned
  end

  User(["Domain user"])

  Repo --> CI --> Train --> Deploy
  Sources --> DataPipe --> FStore
  FStore --> Train
  ExpPipe --> Meta
  Train --> Meta
  Train --> Reg
  Reg --> Deploy --> Serving
  User -- "request" --> Serving -- "response" --> User
  Serving -.-> Shadow
  Serving -.-> Log
  Shadow -.-> Log
  Log -.-> Monitor
  Monitor -.-> CT
  CT -. "retraining order" .-> Train
```

## 2. Lifecycle flow

```mermaid
flowchart LR
  classDef planned stroke-dasharray: 5 5,color:#666

  Raw[("Raw era files<br/>DVC")] --> Pipe["Data pipeline<br/>schema check, cleaning,<br/>features, temporal split"]
  Pipe --> Splits[("Processed splits<br/>data/processed.dvc")]
  Splits --> Cand["Train candidates<br/>baseline, Ridge, RF,<br/>LightGBM, XGBoost"]
  Cand --> Select["Select best on<br/>validation MAE"]
  Select --> Gate{"Promotion gate<br/>vs. live model"}
  Gate -- "better" --> Promote["Push model with DVC,<br/>register in MLflow"]
  Gate -- "not better" --> Keep["Keep live model"]
  Promote --> Image["Docker image<br/>model inside"]
  Keep --> Image
  Image --> Live["Render service"]
  Live --> Users(["Users"])
  Live -.-> Logs[("Prediction log")]:::planned
  Logs -.-> Drift["Drift and shadow<br/>monitoring"]:::planned
  Drift -. "drift" .-> Cand
```

## 3. CI/CD pipeline

```mermaid
flowchart TB
  PR["Pull request"] --> CI1["CI: format, lint, tests,<br/>build image with test model,<br/>container smoke test"]
  CI1 --> Merge{"Review and merge<br/>into main"}
  Merge --> CI2["CI (same checks on main)"]
  Manual["Manual or retraining<br/>trigger"] --> CI2
  CI2 --> Pull["dvc pull<br/>training data + live model"]
  Pull --> TrainAll["Train all candidates<br/>log runs to MLflow"]
  TrainAll --> Gate{"Promotion gate"}
  Gate -- "PROMOTE" --> Save["dvc push new model<br/>commit models.dvc<br/>register version"]
  Gate -- "KEEP" --> Same["Live model unchanged"]
  Save --> Build["Build image<br/>tag = commit SHA"]
  Same --> Build
  Build --> Push["Push to GitHub<br/>Container Registry"]
  Push --> Hook["Render deploy hook<br/>with image tag"]
  Hook --> Wait{"/health reports<br/>new commit?"}
  Wait -- "yes" --> Smoke["Smoke test live site<br/>/, /model/info, /predict"]
  Wait -- "no, 10 min" --> Fail["Pipeline fails,<br/>previous version stays live"]
  Rollback["Rollback workflow<br/>(manual, by commit SHA)"] --> Hook
```

## 4. Deployment view

```mermaid
flowchart LR
  classDef planned stroke-dasharray: 5 5,color:#666

  Dev(["Developer"]) -- "git push / PR" --> GH["GitHub<br/>code, .dvc pointers"]
  GH --> Runner["GitHub Actions runner<br/>(temporary Linux VM)"]
  Runner <-- "dvc pull / push" --> DVC[("DagsHub<br/>DVC storage")]
  Runner -- "log runs, register" --> MLF[("DagsHub<br/>MLflow")]
  Runner -- "push image" --> GHCR[("GitHub Container<br/>Registry")]
  Runner -- "deploy hook" --> Render["Render web service<br/>Docker container"]
  GHCR -- "pull image" --> Render
  User(["User browser"]) -- "HTTPS" --> Render
  Render -.-> PG[("Neon Postgres<br/>prediction log")]:::planned
  PG -.-> Graf["Grafana Cloud<br/>dashboard"]:::planned
  PG -.-> Runner
```

The raw data, processed splits, trained models and experiment records are not
committed to Git. Git holds the code and the DVC pointers that identify the
exact data and model versions.

## 5. Container view

One Docker container holds the frontend, the API and the model. See
[ADR 001](decisions/001-single-container-deployment.md).

```mermaid
flowchart LR
  User["User<br/>(web browser)"]
  subgraph Host["Render web service"]
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

## 6. Prediction request sequence

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

## 7. Training and promotion sequence

What happens in the CD pipeline after a merge into `main`. See
[ADR 003](decisions/003-training-pipeline-and-promotion-gate.md).

```mermaid
sequenceDiagram
  participant GH as GitHub (main)
  participant GA as GitHub Actions
  participant DVC as DagsHub DVC
  participant ML as DagsHub MLflow
  participant REG as GitHub Container Registry
  participant R as Render

  GH->>GA: Push to main (merged PR)
  GA->>GA: CI: lint, tests, container smoke test
  GA->>DVC: dvc pull data/processed.dvc models.dvc
  DVC-->>GA: Training splits + live model
  loop Each candidate (baseline, Ridge, RF, LightGBM, XGBoost)
    GA->>GA: Train on train split, score on validation
    GA->>ML: Log parameters, metrics, model
  end
  GA->>GA: Challenger = lowest validation MAE
  GA->>GA: Promotion gate: challenger vs. live model on test split
  alt Challenger promoted
    GA->>DVC: dvc push new model
    GA->>GH: Commit updated models.dvc
    GA->>ML: Register new version of birth-weight-model
  else Challenger rejected
    GA->>GA: Keep live model, write reasons to job summary
  end
  GA->>DVC: dvc pull models.dvc (model to deploy)
  GA->>REG: Push image tagged with commit SHA
  GA->>R: Deploy hook with image tag
  loop Until live or 10 minutes
    GA->>R: GET /health
    R-->>GA: version, model_ready
  end
  GA->>R: Smoke test /, /model/info, /predict
```

## 8. Analytical pipeline architecture

```mermaid
flowchart LR
  Splits[("train / val / test<br/>Parquet")] --> XY["get_X_y()<br/>21 model features,<br/>target weight_grams"]
  XY --> Pre["Preprocessor<br/>median imputation (numeric),<br/>most-frequent (binary),<br/>scaling for Ridge"]
  Pre --> C1["dummy_mean<br/>(baseline)"]
  Pre --> C2["Ridge"]
  Pre --> C3["Random Forest"]
  Pre --> C4["LightGBM"]
  Pre --> C5["XGBoost"]
  C1 & C2 & C3 & C4 & C5 --> Val["Validation MAE,<br/>RMSE, R²"]
  Val --> Best["select_best()<br/>lowest validation MAE"]
  Best --> Test["Test metrics +<br/>subgroups"]
  Test --> Out["birth_weight_pipeline.joblib<br/>model_metadata.json"]
  C1 & C2 & C3 & C4 & C5 -.-> MLF[("MLflow runs")]
```
