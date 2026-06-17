# ✅ End-to-End MLOps Project Checklist
## Weight Analysis of Newborns — STP Natality

> **How to use this checklist:**  
> Mark each item with `[x]` when complete. Add notes or linked artifacts inline.  
> This checklist covers all 7 phases of the MLOps-based DS process model.

---

## 📊 Progress Overview

| Phase | Status | Owner | Due |
|-------|--------|-------|-----|
| 1. Business Understanding | ✅ Complete | PM / DS Translator | Week 2 |
| 2. Data Engineering | ✅ Complete | Data Engineer | Week 5 |
| 3. Modeling | ⬜ Not Started | ML Engineer | Week 9 |
| 4. Evaluation | ⬜ Not Started | Full Team | Week 11 |
| 5. Deployment | ⬜ Not Started | MLOps Engineer | Week 15 |
| 6. Utilization & Monitoring | ⬜ Not Started | MLOps Engineer | Week 18 |
| 7. Documentation & Reflection | ⬜ Not Started | Full Team | Week 20 |

---

---

## 🏢 Phase 1: Business Understanding

> **Goal:** Establish a shared, unambiguous understanding of the problem, define measurable success criteria, and assess project feasibility.

### 1.1 Problem Conceptualization & Scoping
- [x] Define the business problem in precise, domain-specific terms
  - *"Predict the birth weight (in grams) of newborns from prenatal and demographic data to assist physicians in identifying underweight/overweight risk."*
- [x] Identify the primary **stakeholders** and their roles
  - [x] Healthcare providers (primary end-users)
  - [x] Hospital data custodians
  - [x] Research advisors (Prof. Dr. Turowski, M. Sc. Haertel)
- [x] Formulate the **DS problem type**: Supervised Regression
- [x] Identify the **prediction target**: Birth weight (continuous, in grams)
- [x] Define **clinical relevance thresholds**:
  - [x] Low birth weight (LBW): < 2,500g
  - [x] Very low birth weight (VLBW): < 1,500g
  - [x] Macrosomia: > 4,000g
- [x] Document the **impact of wrong predictions** (false negatives for LBW)

### 1.2 Success Criteria Definition
- [x] Define **ML performance metrics** (primary + secondary):
  - [x] Mean Absolute Error (MAE) — primary metric; target: < 300g
  - [x] Root Mean Squared Error (RMSE) — penalizes large errors
  - [x] R² Score — overall variance explained
  - [x] Clinical Sensitivity/Specificity for LBW classification threshold
- [x] Define **business success criteria**:
  - [x] System reduces undetected LBW cases in test simulation
  - [x] Prediction latency < 500ms per request
  - [x] System uptime > 99% in monitoring period
- [x] Get stakeholder sign-off on acceptance criteria

### 1.3 Feasibility Assessment
- [x] Assess **data availability**: Is the historical dataset sufficient?
  - [x] Minimum sample size analysis (statistical power)
  - [x] Temporal coverage of the dataset
- [x] Assess **ethical and legal considerations**:
  - [x] Data anonymization/de-identification status
  - [x] GDPR compliance considerations
  - [x] Institutional approval for data use
- [x] Assess **technical feasibility**: team skills, infrastructure requirements
- [x] Conduct initial **literature review**:
  - [x] Review existing birth weight prediction models and studies
  - [x] Review MLOps-based DS process model (CRISP-ML(DM) or equivalent)
  - [x] Identify key features commonly used in birth weight prediction

### 1.4 Project Planning
- [x] Create **project charter** document
- [x] Define **team roles & responsibilities** (see [project_plan.md](./docs/project_plan.md))
- [x] Set up **communication channels** (e.g., Teams, Slack)
- [x] Create **project timeline** with milestones (5-month schedule)
- [x] Set up **Git repository** with branch strategy (e.g., GitFlow)
- [x] Configure **project management board** (GitHub Projects / Jira / Trello)
- [x] Schedule **regular sync meetings** (e.g., weekly sprint reviews)

### 1.5 Environment & Infrastructure Setup
- [x] Initialize Git repository and `.gitignore`
- [x] Set up Python virtual environment (`environment.yml` or `requirements.txt`)
- [ ] Install and configure **MLflow** tracking server *(Phase 3)*
- [x] Install and initialize **DVC** for data versioning
- [ ] Set up shared remote storage for DVC (e.g., Google Drive, S3, or local NAS) *(configure when team shares access)*
- [x] Verify all team members can clone repo and run the environment

**📄 Deliverable:** `docs/business_understanding.md` — Problem statement, KPIs, feasibility analysis ✅

---

---

## 🗄️ Phase 2: Data Engineering (Collection, Exploration & Preparation)

> **Goal:** Acquire, understand, validate, and transform the raw dataset into a clean, reproducible data pipeline ready for modeling.

### 2.1 Data Collection & Ingestion
- [x] Receive and securely store the raw dataset in `newborn_data/`
- [x] Version the raw dataset with **DVC**:
  ```bash
  dvc add newborn_data/<dataset_file>
  git add newborn_data/<dataset_file>.dvc .gitignore
  git commit -m "feat: add raw natality dataset"
  ```
- [x] Document **data provenance**:
  - [x] Source of the dataset (NCHS Natality Public Use Files)
  - [x] Number of records and features (26,968,935 rows × 26 columns)
  - [x] Collection methodology (US birth certificate registrations, 1986–2021)
  - [x] Any known data collection biases (50% sample areas, availability gaps by year)
- [x] Identify and document the **data schema** (all columns, types, units) — see `docs/data_documentation.md`
- [x] Check for and handle **data access restrictions** (data is de-identified, no PII)

### 2.2 Exploratory Data Analysis (EDA)
- [x] Open notebook: `notebooks/02_data_exploration.ipynb`
- [x] Compute **basic statistics** for all features (mean, std, min, max, quartiles)
- [x] Visualize **target variable distribution** (birth weight):
  - [x] Histogram / KDE plot
  - [x] Check for normality (Q-Q plot — slight left skew confirmed)
  - [x] Identify outliers (clinical range [300g, 7000g] applied)
- [x] Analyze **feature distributions**:
  - [x] Histograms for numerical features
  - [x] Bar charts for categorical features
  - [x] Box plots grouped by target (LBW vs normal)
- [x] Assess **missing values**:
  - [x] Percentage of missing values per column
  - [x] Missing value patterns (structured/MNAR by year for substance use columns)
  - [x] Visualization (missingno matrix)
- [x] Analyze **correlations**:
  - [x] Pearson correlation matrix (numerical features)
  - [x] Point-biserial correlation (binary features vs. target)
  - [x] Ranked feature correlations with birth weight
- [x] Identify **likely predictive features** (gestation_weeks r=0.38 strongest)
- [x] Detect potential **data leakage** (apgar_1min, apgar_5min confirmed leakage — excluded)
- [x] Analyze **temporal trends** (if data is multi-year):
  - [x] Subtle downward trend in mean birth weight (~20g over 35 years)
  - [x] LBW prevalence slightly increasing in recent eras
- [x] Document key EDA findings and initial hypotheses — see `docs/data_report.md`

### 2.3 Data Validation
- [x] Set up **data validation test suite** (`tests/data_validation/test_data_expectations.py`):
  - *Implemented as pytest-based assertions (GE 0.18+ API breaking changes made full GE setup impractical)*
- [x] Define **data expectations / constraints**:
  - [x] Birth weight range: [300g, 7000g]
  - [x] Gestational age range: [20, 44] weeks
  - [x] Non-null constraints for `weight_grams`
  - [x] Valid binary values for `is_male`, `mother_married`, flag columns
  - [x] Valid `plurality` range [1, 9]
- [x] Run validation suite on processed splits (66 tests pass after pipeline execution)
- [x] Validation runs automatically: `pytest tests/data_validation/ -v`

### 2.4 Data Preparation & Preprocessing
- [x] Open notebook: `notebooks/03_data_preparation.ipynb`
- [x] **Handle missing values**:
  - [x] `father_age=99`, `gestation_weeks=99`, `lmp∈{"99","9999"}` → NaN (sentinel replacement)
  - [x] High-missingness columns (alcohol_use, cigarette_use) retained with NaN — imputation deferred to Phase 3
- [x] **Handle outliers**:
  - [x] `gestation_weeks` clipped to [20, 44] — values >44 are data errors
  - [x] `mother_age` clipped to [10, 60]
  - [x] Birth weight rows outside [300g, 7000g] removed (~0.02% of data)
  - [x] `outlier_weight` flag column created
- [x] **Encode categorical variables**:
  - [x] Boolean strings ("true"/"false") → Int8 (0/1)
  - [x] Mother age ordinal binning: 0=teen, 1=20s, 2=30s, 3=40+
  - [x] Geographic high-cardinality columns retained (encoding deferred to Phase 3 with training data only)
- [x] **Feature engineering**:
  - [x] `weight_gain_kg` (kg conversion), `parity` (prior births), `is_multiple_birth`, `lmp_known`
  - [x] `birth_month_sin`/`birth_month_cos` (cyclical encoding)
  - [x] `gestation_preterm`, `gestation_post_term` (clinical threshold flags)
- [x] **Train/Validation/Test split**:
  - [x] Strategy: **temporal (year-based)** split — avoids future-data leakage
  - [x] Train: < 2010 (~70%) | Val: 2010–2015 (~15%) | Test: ≥ 2016 (~15%)
  - [x] No temporal overlap verified
  - [x] LBW rate stable across splits (~7%)
- [x] Save processed datasets to `data/processed/` (Parquet format)
- [x] Build a **reproducible preprocessing pipeline** in `src/data/`:
  - [x] `ingest.py` — raw data loading
  - [x] `preprocess.py` — cleaning and transformation
  - [x] `feature_engineering.py` — derived features
  - [x] `split.py` — data splitting logic
- [x] `scripts/run_pipeline.py` — standalone CLI pipeline runner

**📄 Deliverable:** `docs/data_report.md` — EDA findings, data quality report, preprocessing decisions ✅

---

---

## 🤖 Phase 3: Modeling

> **Goal:** Select, train, tune, and evaluate ML models using experiment tracking. Identify the best model for production.

### 3.1 Baseline Model
- [ ] Open notebook: `notebooks/04_modeling.ipynb`
- [ ] Train a **baseline model** (simple rule or DummyRegressor using mean prediction)
  - [ ] Record baseline MAE, RMSE, R² on validation set
  - [ ] Log baseline experiment to **MLflow**:
    ```python
    mlflow.start_run(run_name="baseline_dummy")
    mlflow.log_metric("mae", baseline_mae)
    ```
- [ ] Establish baseline as the **minimum performance bar**

### 3.2 Candidate Model Selection
- [ ] Select a set of candidate models to compare:
  - [ ] **Linear Regression** (interpretable baseline)
  - [ ] **Ridge / Lasso Regression** (regularization, feature selection)
  - [ ] **Decision Tree Regressor** (non-linear, interpretable)
  - [ ] **Random Forest Regressor** (ensemble, robust)
  - [ ] **Gradient Boosting: XGBoost / LightGBM** (high-performance, likely best)
  - [ ] **Support Vector Regression (SVR)** (optional)
  - [ ] **Neural Network / MLP** (optional, if time permits)
- [ ] Document rationale for each model type selected
- [ ] Implement a **cross-validation** strategy (k-fold, k=5 or k=10)

### 3.3 Initial Model Training & Comparison
- [ ] Train each candidate model with default hyperparameters
- [ ] Log each experiment to MLflow:
  - [ ] Parameters (model type, key hyperparameters)
  - [ ] Metrics (MAE, RMSE, R² on train and validation)
  - [ ] Artifacts (model file, feature importance plots)
- [ ] Compare models in MLflow UI (`http://localhost:5000`)
- [ ] Create **comparison table** of all candidate models
- [ ] Shortlist **top 2-3 models** for hyperparameter tuning

### 3.4 Hyperparameter Optimization (HPO)
- [ ] Choose HPO strategy:
  - [ ] **Grid Search** (small search space)
  - [ ] **Random Search** (medium search space)
  - [ ] **Bayesian Optimization** (Optuna — recommended for efficiency)
- [ ] Define hyperparameter search space for each shortlisted model
- [ ] Run HPO with cross-validation on training set (NEVER on test set)
- [ ] Log all HPO trials to MLflow
- [ ] Identify best hyperparameters for each model
- [ ] Retrain final model with best hyperparameters on full training set

### 3.5 Feature Importance & Interpretability
- [ ] Compute **feature importance** for the best model:
  - [ ] Built-in feature importance (tree-based models)
  - [ ] Permutation importance (model-agnostic)
  - [ ] SHAP values (SHapley Additive exPlanations) — strongly recommended
- [ ] Identify the top N most predictive features
- [ ] Validate feature importance makes clinical sense
- [ ] Document surprising or counter-intuitive findings

### 3.6 Model Evaluation on Test Set
- [ ] Evaluate the **selected best model** on the held-out test set (ONCE):
  - [ ] MAE, RMSE, R² scores
  - [ ] Residual analysis (plot residuals vs. predicted, check for patterns)
  - [ ] Error distribution plot
- [ ] Evaluate **clinical performance**:
  - [ ] Classify predictions as LBW (<2500g) / Normal / Macrosomia (>4000g)
  - [ ] Compute confusion matrix, sensitivity, specificity for LBW detection
- [ ] Check for **subgroup fairness** (by sex, gestational age bracket, year)
- [ ] Check **robustness** (performance on different years of data)
- [ ] Compare against baseline — confirm significant improvement
- [ ] Log final model to **MLflow Model Registry**:
  ```python
  mlflow.register_model(model_uri, "NewbornWeightPredictor")
  ```

### 3.7 Model Versioning & Packaging
- [ ] Register final model in MLflow Model Registry with stage: `Staging`
- [ ] Save model artifacts: serialized model, scaler, feature list, metadata
- [ ] Write a `model_card.md` documenting:
  - [ ] Model type and version
  - [ ] Training data description
  - [ ] Performance metrics
  - [ ] Known limitations and biases
  - [ ] Intended use and out-of-scope uses

**📄 Deliverable:** `docs/modeling_report.md` — Model selection rationale, HPO results, evaluation metrics, model card

---

---

## 📈 Phase 4: Evaluation (Business Validation)

> **Goal:** Assess the model against the original business goals and obtain formal project sign-off before deployment.

### 4.1 Business Goal Assessment
- [ ] Review the success criteria defined in Phase 1
- [ ] Map each ML metric to a business outcome:
  - [ ] MAE < 300g → clinically acceptable precision
  - [ ] LBW sensitivity > 85% → acceptable safety level
- [ ] Assess whether the model meets **all defined acceptance criteria**
- [ ] Document areas where the model falls short (if any) and mitigation strategy

### 4.2 Risk & Ethical Assessment
- [ ] Review model predictions for **demographic bias**:
  - [ ] Is performance consistent across maternal age groups?
  - [ ] Is performance consistent across years of data?
- [ ] Document any identified biases and their clinical implications
- [ ] Assess **model uncertainty**: Does the model flag low-confidence predictions?
- [ ] Define the **human-in-the-loop** workflow:
  - [ ] The model is a **decision support tool** — doctors always have final authority
  - [ ] Define when to escalate to a specialist (e.g., prediction near LBW threshold)
- [ ] Review for GDPR compliance in the deployment context

### 4.3 Go/No-Go Decision
- [ ] Conduct a formal **evaluation review meeting** with team + advisors
- [ ] Present evaluation findings (prepared slide deck or report)
- [ ] Obtain formal **Go/No-Go sign-off** for deployment:
  - [ ] ✅ Go: model meets all criteria → proceed to Phase 5
  - [ ] ⚠️ Conditional Go: minor issues to fix in parallel → document conditions
  - [ ] 🚫 No-Go: return to Phase 3 with documented findings
- [ ] Document the Go/No-Go decision and rationale

**📄 Deliverable:** Evaluation review presentation + updated `docs/modeling_report.md`

---

---

## 🚀 Phase 5: Deployment

> **Goal:** Build, test, and deploy the ML-powered prediction service in a reproducible, scalable, and maintainable way.

### 5.1 Prediction Service Development
- [ ] Design the **REST API** architecture (FastAPI):
  - [ ] `POST /predict` — accepts features, returns predicted birth weight
  - [ ] `GET /health` — health check endpoint
  - [ ] `GET /model/info` — model version and metadata
- [ ] Implement `src/api/main.py`:
  - [ ] Load model from MLflow Model Registry
  - [ ] Implement input validation (Pydantic schema)
  - [ ] Implement prediction logic and output formatting
  - [ ] Add error handling and logging
- [ ] Define the **API input schema** (request body):
  ```json
  {
    "gestational_age_weeks": 39,
    "maternal_age": 28,
    "maternal_bmi": 24.5,
    "parity": 1,
    "sex": "M",
    ...
  }
  ```
- [ ] Define the **API output schema** (response body):
  ```json
  {
    "predicted_weight_grams": 3412,
    "risk_category": "normal",
    "confidence_note": "High confidence prediction",
    "model_version": "1.0.3"
  }
  ```

### 5.2 Containerization
- [ ] Create `deployment/Dockerfile`:
  - [ ] Base image: `python:3.11-slim`
  - [ ] Install dependencies from `requirements.txt`
  - [ ] Copy application code
  - [ ] Expose port 8000
  - [ ] Set entrypoint: `uvicorn src.api.main:app`
- [ ] Create `deployment/docker-compose.yml`:
  - [ ] `api` service (prediction API)
  - [ ] `mlflow` service (model registry)
  - [ ] `prometheus` service (metrics scraping)
  - [ ] `grafana` service (visualization dashboards)
- [ ] Build and test Docker image locally:
  ```bash
  docker build -t stp-natality-api:latest .
  docker run -p 8000:8000 stp-natality-api:latest
  ```
- [ ] Verify API responds correctly inside container

### 5.3 Testing
- [ ] Write **unit tests** (`tests/unit/`):
  - [ ] Test preprocessing functions with known inputs/outputs
  - [ ] Test feature engineering logic
  - [ ] Test model loading and prediction format
- [ ] Write **integration tests** (`tests/integration/`):
  - [ ] Test API endpoints (happy path, edge cases, invalid inputs)
  - [ ] Test API response schema matches specification
  - [ ] Test health check endpoint
- [ ] Write **data validation tests** (`tests/data_validation/`):
  - [ ] Run Great Expectations test suite on processed data
- [ ] Set up **test coverage reporting** (aim for > 80% coverage):
  ```bash
  pytest --cov=src tests/
  ```
- [ ] All tests must pass before deployment

### 5.4 CI/CD Pipeline
- [ ] Set up **GitHub Actions** workflow (`.github/workflows/ci.yml`):
  - [ ] Trigger: on push to `main` and `develop` branches, on pull requests
  - [ ] Steps:
    - [ ] Checkout code
    - [ ] Set up Python environment
    - [ ] Install dependencies
    - [ ] Run linting (flake8, black --check)
    - [ ] Run all tests with coverage
    - [ ] Build Docker image
    - [ ] (Optional) Push Docker image to registry (Docker Hub / GitHub Container Registry)
- [ ] Create a **CD pipeline** for deployment:
  - [ ] Automated deployment to staging environment on `develop` merge
  - [ ] Manual approval gate for production deployment

### 5.5 Deployment Execution
- [ ] Deploy to **staging environment**:
  - [ ] Run full test suite against staging deployment
  - [ ] Perform manual smoke tests
  - [ ] Verify monitoring services are running
- [ ] Conduct **User Acceptance Testing (UAT)**:
  - [ ] Demonstrate prediction API to advisors
  - [ ] Gather feedback and address issues
- [ ] Deploy to **production environment** (or final demo environment):
  - [ ] Update MLflow model stage: `Staging` → `Production`
  - [ ] Document deployment steps in `docs/deployment_guide.md`

**📄 Deliverable:** Deployed API + `docs/deployment_guide.md` — architecture, deployment runbook, rollback procedure

---

---

## 📡 Phase 6: Utilization & Monitoring

> **Goal:** Demonstrate the running system, monitor its performance in production, and establish maintenance processes.

### 6.1 System Demonstration
- [ ] Prepare a **live demonstration script**:
  - [ ] Show API call via UI / Postman / curl
  - [ ] Show prediction results for sample cases (including LBW edge cases)
  - [ ] Show MLflow model registry and experiment logs
  - [ ] Show monitoring dashboards
- [ ] Record a **demonstration video** or conduct a live session with advisors

### 6.2 Monitoring Setup
- [ ] Configure **Prometheus** to scrape API metrics:
  - [ ] Request count, latency, error rate
  - [ ] Prediction value distribution
- [ ] Set up **Grafana dashboards** (`monitoring/dashboards/`):
  - [ ] API performance dashboard (latency, throughput, errors)
  - [ ] Prediction distribution dashboard (distribution of predicted weights over time)
  - [ ] Data drift dashboard
- [ ] Implement **Evidently AI** for ML-specific monitoring:
  - [ ] Data drift report (input feature distributions)
  - [ ] Model performance report (if ground truth becomes available)
  - [ ] Target drift report (shift in predicted weight distribution)
- [ ] Configure **alerting rules** (`monitoring/alerts/`):
  - [ ] Alert: API error rate > 5%
  - [ ] Alert: Average latency > 1000ms
  - [ ] Alert: Significant data drift detected (PSI or KS statistic threshold)
  - [ ] Alert: Predicted weight distribution shift

### 6.3 Drift Detection & Retraining Strategy
- [ ] Define **drift detection thresholds** for each key feature
- [ ] Define **model retraining triggers**:
  - [ ] Scheduled retraining (e.g., every 6 months with new data)
  - [ ] Performance-based trigger (MAE degrades by > X% on new labeled data)
  - [ ] Drift-based trigger (PSI > 0.2 for key features)
- [ ] Document the **retraining procedure**:
  - [ ] Data collection → validation → retraining → evaluation → staging → production
  - [ ] How to compare new model vs. current production model
- [ ] Implement a **feedback loop** mechanism (if applicable):
  - [ ] Process for doctors to flag incorrect predictions
  - [ ] How flagged cases are incorporated into retraining data

### 6.4 Maintenance Processes
- [ ] Document **system maintenance procedures**:
  - [ ] How to update the model (MLflow model promotion)
  - [ ] How to roll back to a previous model version
  - [ ] How to update dependencies and Docker image
  - [ ] Backup and recovery procedures
- [ ] Define **on-call runbook** for common issues:
  - [ ] API is down → restart Docker container
  - [ ] MLflow server unavailable → fallback to cached model
  - [ ] Data drift alert → trigger manual model review

**📄 Deliverable:** `docs/monitoring_guide.md` — monitoring setup, drift detection strategy, maintenance runbook

---

---

## 📝 Phase 7: Documentation & Critical Reflection

> **Goal:** Produce comprehensive, scientifically robust documentation of all implementation steps and critically reflect on the project.

### 7.1 Technical Documentation
- [ ] Ensure all code has **docstrings** and **inline comments**
- [ ] Generate **API documentation** (FastAPI auto-generates Swagger UI at `/docs`)
- [ ] Verify `README.md` is complete and up to date
- [ ] Complete all phase-specific documentation files:
  - [ ] `docs/business_understanding.md` ✅
  - [ ] `docs/data_report.md` ✅
  - [ ] `docs/modeling_report.md` ✅
  - [ ] `docs/deployment_guide.md` ✅
  - [ ] `docs/monitoring_guide.md` ✅

### 7.2 Architectural Diagrams
- [ ] Create **System Architecture Diagram** (overall MLOps system):
  - [ ] Data sources → preprocessing pipeline → model training → model registry → API → monitoring
  - [ ] Tool: draw.io, PlantUML, Lucidchart, or Mermaid
- [ ] Create **Data Pipeline Diagram**:
  - [ ] Raw data → validation → preprocessing → feature engineering → train/val/test splits
- [ ] Create **CI/CD Pipeline Diagram**:
  - [ ] Code push → lint → test → build → deploy to staging → manual gate → production
- [ ] Create **Monitoring Architecture Diagram**:
  - [ ] API → Prometheus → Grafana + Evidently AI → alerting
- [ ] Save all diagrams to `docs/diagrams/`

### 7.3 Final Scientific Report
- [ ] Draft `docs/final_report.md` following scientific reporting standards:
  - [ ] **Abstract** — project summary and key results
  - [ ] **1. Introduction** — problem motivation, healthcare context, project objectives
  - [ ] **2. Related Work** — review of birth weight prediction literature, MLOps frameworks
  - [ ] **3. Methodology** — the MLOps-based DS process model applied
  - [ ] **4. Business Understanding** — problem formulation, KPIs, feasibility
  - [ ] **5. Data** — dataset description, EDA findings, quality issues, preprocessing
  - [ ] **6. Modeling** — model selection, training, evaluation results (with tables and figures)
  - [ ] **7. Deployment** — system architecture, API design, CI/CD setup
  - [ ] **8. Monitoring & Utilization** — monitoring approach, drift strategy, maintenance
  - [ ] **9. Critical Reflection** — (see 7.4)
  - [ ] **10. Conclusion** — achieved objectives, clinical relevance, future work
  - [ ] **References** — properly cited academic sources
- [ ] Include **tables and figures** for all key results
- [ ] Ensure **reproducibility**: all steps are sufficiently documented to replicate

### 7.4 Critical Reflection
- [ ] Reflect on the **MLOps-based process model**:
  - [ ] Which phases were most valuable? Most challenging?
  - [ ] Were all phases followed strictly? If not, why?
  - [ ] Would a different process model have been more suitable?
- [ ] Reflect on **used technologies**:
  - [ ] What worked well? (e.g., MLflow for experiment tracking)
  - [ ] What would you replace? (e.g., choose Optuna over GridSearch earlier)
  - [ ] Unexpected technical challenges encountered
- [ ] Reflect on **data challenges**:
  - [ ] Data quality issues discovered
  - [ ] Impact of missing data on model performance
  - [ ] Data limitations (collection bias, historical shifts)
- [ ] Reflect on **team and project management**:
  - [ ] How well did roles and responsibilities work?
  - [ ] Communication challenges and solutions
  - [ ] What you would do differently in a future project
- [ ] Reflect on **clinical and ethical implications**:
  - [ ] Limitations of AI in clinical decision-making
  - [ ] Importance of the human-in-the-loop approach
  - [ ] Potential for bias and how it was addressed

### 7.5 Final Review & Submission
- [ ] Internal team review of the final report (peer review all sections)
- [ ] Technical review: verify all code runs end-to-end from a clean environment
- [ ] Advisor review: share draft report with M. Sc. Haertel for feedback
- [ ] Incorporate feedback and finalize report
- [ ] Final code cleanup:
  - [ ] Remove debug code and temporary notebooks
  - [ ] Ensure all notebooks have been run with clean outputs
  - [ ] Tag the final release in Git: `git tag v1.0.0`
- [ ] Submit all deliverables:
  - [ ] 🗂️ Code repository (GitHub link)
  - [ ] 📄 Final scientific report
  - [ ] 🎬 Demonstration video / presentation
  - [ ] ✅ This completed checklist

---

---

## 🏆 Final Deliverables Summary

| # | Deliverable | Status | Location |
|---|-------------|--------|----------|
| 1 | Business Understanding Document | ⬜ | `docs/business_understanding.md` |
| 2 | Data Report (EDA + Preprocessing) | ⬜ | `docs/data_report.md` |
| 3 | Modeling Report + Model Card | ⬜ | `docs/modeling_report.md` |
| 4 | Deployment Guide + Runbook | ⬜ | `docs/deployment_guide.md` |
| 5 | Monitoring Guide | ⬜ | `docs/monitoring_guide.md` |
| 6 | Architectural Diagrams | ⬜ | `docs/diagrams/` |
| 7 | Final Scientific Report | ⬜ | `docs/final_report.md` |
| 8 | Complete Code Repository | ⬜ | GitHub |
| 9 | Deployed Prediction API | ⬜ | Docker / Server |
| 10 | Demonstration (video / live) | ⬜ | TBD |

---

*Last updated: May 2026 | STP Natality Project*
