# 🍼 Development and Operationalization of Data Science Solutions
## Weight Analysis of Newborns

> **Institute:** Institut für Technische und Betriebliche Informationssysteme  
> **Advisors:** Prof. Dr. Klaus Turowski, M. Sc. Christian Haertel  
> **Team Size:** 4 members | **Duration:** 5 months  
> **Knowledge Required:** Python, Data Analysis, Data Engineering, Machine Learning, Software Engineering, Technical Writing

---

## 📋 Project Overview

This project implements an **end-to-end Data Science (DS) and MLOps solution** to predict the birth weight of newborns using historical natality data. The prediction system is intended to assist healthcare providers — particularly during routine prenatal examinations — in identifying potential health risks such as underweight or overweight newborns.

The project follows an **MLOps-based DS process model**, covering the full lifecycle from business understanding through data engineering, modeling, deployment, and production monitoring.

---

## 🎯 Project Goal

> **Predict the birth weight of newborns** based on historical data to help doctors proactively identify and respond to potential health issues.

**Primary ML Task:** Regression (continuous target: birth weight in grams)  
**Success Metric:** Model achieves clinically meaningful accuracy (e.g., MAE < 300g) with a deployable, monitored production system.

---

## 🗂️ Directory Structure

```
STP Natality/
│
├── README.md                          # ← You are here: Project overview
├── checklist.md                       # End-to-end MLOps project checklist
│
├── data/
│   ├── raw/                           # Raw unprocessed dataset (as received)
│   ├── processed/                     # Cleaned and feature-engineered data
│   └── external/                      # Any supplementary or reference data
│
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_monitoring_analysis.ipynb
│
├── src/
│   ├── data/                          # Data ingestion & preprocessing scripts
│   ├── features/                      # Feature engineering pipeline
│   ├── models/                        # Model training, evaluation, registry
│   ├── api/                           # FastAPI prediction service
│   └── monitoring/                    # Drift detection & monitoring scripts
│
├── models/                            # Serialized model artifacts (tracked by DVC/MLflow)
│
├── tests/
│   ├── unit/                          # Unit tests for src/ modules
│   ├── integration/                   # Integration tests for the API
│   └── data_validation/              # Great Expectations / data quality tests
│
├── deployment/
│   ├── Dockerfile                     # Container definition for prediction service
│   ├── docker-compose.yml             # Multi-service orchestration (API + monitoring)
│   └── ci_cd/                         # GitHub Actions / CI-CD pipeline configs
│
├── monitoring/
│   ├── dashboards/                    # Grafana dashboard definitions
│   └── alerts/                        # Alert rules and thresholds
│
├── docs/
│   ├── project_plan.md               # Detailed project plan, timeline & roles
│   ├── business_understanding.md     # Business problem formulation
│   ├── data_report.md                # Data exploration & quality report
│   ├── modeling_report.md            # Model selection, training & evaluation report
│   ├── deployment_guide.md           # Deployment architecture & runbook
│   ├── monitoring_guide.md           # Monitoring setup and maintenance guide
│   ├── final_report.md               # Comprehensive scientific final report
│   └── diagrams/                     # Architecture and workflow diagrams (draw.io / PlantUML)
│
├── mlruns/                            # MLflow experiment tracking (auto-generated)
├── .dvc/                              # DVC configuration for data versioning
├── .github/workflows/                 # CI/CD pipeline definitions
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment specification
└── .gitignore
```

---

## 🔄 MLOps Process Model Phases

The project follows a structured MLOps lifecycle, as tracked in [checklist.md](./checklist.md):

| # | Phase | Key Deliverable |
|---|-------|----------------|
| 1 | **Business Understanding** | Problem statement, KPIs, feasibility analysis |
| 2 | **Data Engineering** | Clean dataset, feature pipeline, data validation |
| 3 | **Modeling** | Trained & evaluated ML models, MLflow experiment logs |
| 4 | **Evaluation** | Business-aligned model assessment, sign-off |
| 5 | **Deployment** | Dockerized API, CI/CD pipeline, integration tests |
| 6 | **Utilization & Monitoring** | Monitoring dashboards, drift alerts, retraining triggers |
| 7 | **Documentation & Reflection** | Final scientific report, architectural diagrams |

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Data Processing | Pandas, NumPy, Polars |
| EDA & Visualization | Matplotlib, Seaborn, Plotly |
| ML Frameworks | Scikit-learn, XGBoost, LightGBM |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| Data Validation | Great Expectations |
| API / Serving | FastAPI + Uvicorn |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI, Prometheus, Grafana |
| Version Control | Git + GitHub |

---

## 🚀 Quickstart

### 1. Clone the Repository
```bash
git clone <repository-url>
cd "STP Natality"
```

### 2. Set Up the Python Environment
```bash
# Using conda
conda env create -f environment.yml
conda activate stp-natality

# Or using pip
pip install -r requirements.txt
```

### 3. Initialize DVC (Data Versioning)
```bash
dvc init
dvc pull   # Pull data from remote storage (configure remote first)
```

### 4. Start MLflow Tracking Server
```bash
mlflow ui --port 5000
```

### 5. Run the Prediction API
```bash
cd src/api
uvicorn main:app --reload --port 8000
```

### 6. Run Tests
```bash
pytest tests/
```

---

## 📚 Documentation Index

| Document | Description |
|----------|-------------|
| [checklist.md](./checklist.md) | Full end-to-end MLOps project checklist |
| [docs/project_plan.md](./docs/project_plan.md) | Timeline, milestones, roles & responsibilities |
| [docs/business_understanding.md](./docs/business_understanding.md) | Business problem formulation |
| [docs/data_report.md](./docs/data_report.md) | Data exploration and quality report |
| [docs/modeling_report.md](./docs/modeling_report.md) | Modeling decisions and results |
| [docs/deployment_guide.md](./docs/deployment_guide.md) | Deployment architecture and runbook |
| [docs/final_report.md](./docs/final_report.md) | Final scientific report |

---

## 👥 Team

| Role | Responsibilities |
|------|----------------|
| **Project Manager / DS Translator** | Planning, coordination, business alignment, documentation |
| **Data Engineer** | Data ingestion, preprocessing, pipeline, validation |
| **Machine Learning Engineer** | Modeling, experiment tracking, evaluation |
| **MLOps / DevOps Engineer** | Deployment, CI/CD, monitoring, infrastructure |

---

*Last updated: May 2026*
