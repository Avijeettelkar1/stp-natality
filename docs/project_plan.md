# Project Plan: Weight Analysis of Newborns
## Development and Operationalization of Data Science Solutions

> **Institute:** Institut für Technische und Betriebliche Informationssysteme  
> **Advisors:** Prof. Dr. Klaus Turowski, M. Sc. Christian Haertel  
> **Team Size:** 4 | **Duration:** 5 months (≈ 20 weeks)  
> **Start Date:** [To be defined upon project kick-off]

---

## 1. Project Background & Objectives

### 1.1 Problem Statement
Predicting the birth weight of newborns is a clinically significant task. Abnormal birth weight — specifically **low birth weight (LBW, < 2,500g)** and **macrosomia (> 4,000g)** — are linked to serious perinatal complications. Early prediction during prenatal care allows physicians to take proactive measures.

This project builds a **machine learning-powered prediction system** trained on historical natality data. The system will be designed and operationalized following an **MLOps-based Data Science process model**, ensuring it is not just a research prototype but a deployable, monitored, and maintainable solution.

### 1.2 Project Objectives

| # | Objective | Priority |
|---|-----------|----------|
| 1 | Deliver a trained ML model predicting birth weight with MAE < 300g | Critical |
| 2 | Build a deployable REST API wrapping the model | Critical |
| 3 | Implement CI/CD pipelines for automated testing and deployment | High |
| 4 | Set up real-time monitoring and drift detection | High |
| 5 | Produce comprehensive, scientifically robust documentation | Critical |
| 6 | Critically reflect on the MLOps process model and technologies used | High |

---

## 2. Team Structure & Roles

The project team consists of 4 members. Roles are divided to ensure coverage of all MLOps lifecycle stages:

| Role | Primary Responsibilities | Secondary Responsibilities |
|------|--------------------------|---------------------------|
| **Project Manager (PM) / DS Translator** | Sprint planning, stakeholder communication, business understanding, documentation coordination, final report | Code review, presentation |
| **Data Engineer (DE)** | Data ingestion, EDA, preprocessing pipeline, data validation, DVC setup | Feature engineering support |
| **Machine Learning Engineer (MLE)** | Model selection, training, HPO, evaluation, MLflow tracking, model registry | Feature engineering, API integration |
| **MLOps / DevOps Engineer (MOE)** | Dockerization, CI/CD pipeline, deployment, monitoring setup (Prometheus/Grafana/Evidently) | Testing infrastructure |

> **Note:** In a team of 4, roles may overlap. All team members contribute to documentation and testing.

---

## 3. Technology Stack

### 3.1 Core Stack

| Layer | Technology | Purpose | Version |
|-------|-----------|---------|---------|
| Language | **Python** | All data science and engineering code | 3.11+ |
| Data Processing | **Pandas, NumPy** | Data manipulation and analysis | Latest |
| High-Performance I/O | **Polars** | Optional: faster DataFrame ops | Latest |
| Visualization | **Matplotlib, Seaborn, Plotly** | EDA, reporting charts | Latest |
| ML – Classical | **Scikit-learn** | Preprocessing pipelines, baseline models, SVR, RF | 1.4+ |
| ML – Boosting | **XGBoost, LightGBM** | Primary candidate high-performance models | Latest |
| Interpretability | **SHAP** | Feature importance and model explanation | Latest |
| HPO | **Optuna** | Bayesian hyperparameter optimization | Latest |

### 3.2 MLOps Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Experiment Tracking | **MLflow** | Log parameters, metrics, artifacts; Model Registry |
| Data Versioning | **DVC** | Version datasets and model artifacts alongside Git |
| Data Validation | **Great Expectations** | Automated data quality checks |
| API Framework | **FastAPI + Uvicorn** | Production-ready REST prediction service |
| Containerization | **Docker + Docker Compose** | Reproducible deployment environment |
| CI/CD | **GitHub Actions** | Automated testing, linting, and deployment pipeline |
| Monitoring – Infra | **Prometheus + Grafana** | API latency, throughput, error rate dashboards |
| Monitoring – ML | **Evidently AI** | Data drift, model performance drift reports |
| Testing | **pytest + pytest-cov** | Unit and integration tests with coverage |
| Code Quality | **Black, Flake8, isort** | Code formatting and linting |

### 3.3 Collaboration & Project Management

| Tool | Purpose |
|------|---------|
| **Git + GitHub** | Version control, PR-based workflow |
| **GitHub Projects** | Sprint board and task tracking |
| **GitHub Actions** | CI/CD automation |
| **Markdown + Mermaid / draw.io** | Documentation and architectural diagrams |

---

## 4. Project Timeline & Milestones

The project runs for **5 months (≈ 20 weeks)**, divided into phases. Each phase has a target completion milestone.

### 4.1 Gantt-Style Overview

```
MONTH 1    MONTH 2    MONTH 3    MONTH 4    MONTH 5
|----------|----------|----------|----------|----------|
|Ph1: BU   |                                           |
|    Ph2: Data Eng.   |                               |
|              Ph3: Modeling      |                   |
|                         Ph4: Eval|                  |
|                              Ph5: Deploy            |
|                                   Ph6: Monitor      |
|          Ph7: Documentation (continuous)            |
```

### 4.2 Detailed Phase Timeline

| Phase | Name | Weeks | Key Milestones |
|-------|------|-------|----------------|
| **Phase 1** | Business Understanding | Weeks 1–2 | ✔ Problem statement signed off; ✔ KPIs defined; ✔ Repo & environment set up |
| **Phase 2** | Data Engineering | Weeks 2–5 | ✔ EDA notebook complete; ✔ Data validation suite passing; ✔ Preprocessing pipeline in `src/` |
| **Phase 3** | Modeling | Weeks 5–9 | ✔ Baseline established; ✔ HPO complete; ✔ Best model in MLflow Registry |
| **Phase 4** | Evaluation | Weeks 9–11 | ✔ Business evaluation report; ✔ Go/No-Go decision documented |
| **Phase 5** | Deployment | Weeks 11–15 | ✔ API live (Docker); ✔ CI/CD pipeline passing; ✔ All tests green |
| **Phase 6** | Monitoring & Utilization | Weeks 15–18 | ✔ Grafana dashboards live; ✔ Drift detection configured; ✔ Demo recorded |
| **Phase 7** | Documentation & Reflection | Weeks 1–20 (continuous) | ✔ Final report complete; ✔ Code tagged v1.0.0; ✔ Submission ready |

### 4.3 Sprint Structure (Bi-Weekly Sprints)

| Sprint | Weeks | Focus |
|--------|-------|-------|
| Sprint 1 | 1–2 | Kick-off, literature review, business understanding, environment setup |
| Sprint 2 | 3–4 | Data ingestion, EDA, data validation |
| Sprint 3 | 5–6 | Preprocessing pipeline, feature engineering, baseline model |
| Sprint 4 | 7–8 | Model selection, initial training, MLflow setup |
| Sprint 5 | 9–10 | HPO, model evaluation, business evaluation review |
| Sprint 6 | 11–12 | API development, containerization, unit tests |
| Sprint 7 | 13–14 | CI/CD pipeline, integration tests, staging deployment |
| Sprint 8 | 15–16 | Monitoring setup, drift detection, production deployment |
| Sprint 9 | 17–18 | System demonstration, maintenance docs, monitoring review |
| Sprint 10 | 19–20 | Final report writing, critical reflection, submission |

---

## 5. Risk Management

### 5.1 Risk Register

| # | Risk | Likelihood | Impact | Severity | Mitigation Strategy |
|---|------|-----------|--------|----------|---------------------|
| R1 | **Dataset is insufficient in size or quality** | Medium | High | 🔴 High | Early EDA in Week 2; define minimum requirements; plan for augmentation or feature reduction |
| R2 | **Data contains sensitive PII requiring special handling** | Medium | High | 🔴 High | Check data anonymization status in Week 1; consult advisors on GDPR compliance |
| R3 | **Model fails to meet MAE < 300g target** | Medium | High | 🔴 High | Iterate with HPO; try ensemble methods; consider relaxing threshold with advisor approval |
| R4 | **Team member unavailability / dropout** | Low | High | 🟠 Medium | Cross-train on critical tasks; document all work; maintain updated handover notes |
| R5 | **Deployment infrastructure issues** | Low | Medium | 🟡 Medium | Use Docker for portability; test locally first; use GitHub Actions for automated validation |
| R6 | **Scope creep** | High | Medium | 🟠 Medium | Strict adherence to checklist and sprint goals; PM must enforce scope boundaries |
| R7 | **Significant data drift before project ends** | Low | Low | 🟢 Low | Drift detection is part of Phase 6; document in monitoring guide |
| R8 | **MLflow / DVC configuration issues** | Low | Low | 🟢 Low | Set up in Sprint 1; all team members verify access in Week 2 |

---

## 6. Communication Plan

| Meeting Type | Frequency | Participants | Purpose |
|-------------|-----------|-------------|---------|
| **Sprint Planning** | Every 2 weeks (Monday) | Full team | Plan next sprint tasks, assign items from checklist |
| **Daily Standup** | 3x per week | Full team | Brief progress sync (15 min max): done / doing / blockers |
| **Sprint Review** | Every 2 weeks (Friday) | Full team + advisors (if available) | Demo completed work, gather feedback |
| **Advisor Check-in** | Monthly (or as needed) | PM + 1 team member + advisors | Progress updates, technical guidance |
| **Ad-hoc** | As needed | Relevant members | Resolve blockers, design decisions |

---

## 7. Development Practices

### 7.1 Git Branching Strategy (GitFlow)

```
main        ─── stable, production-ready releases only (tagged)
develop     ─── integration branch for all completed features
feature/*   ─── one branch per feature/task (merged into develop via PR)
hotfix/*    ─── urgent fixes to main
release/*   ─── final stabilization before tagging a release
```

### 7.2 Code Quality Standards
- All code formatted with **Black** (line length: 88)
- All code linted with **Flake8**
- Import order enforced with **isort**
- All functions and classes must have **docstrings** (Google style)
- Minimum **80% test coverage** for `src/` modules
- No direct commits to `main` — all changes via Pull Requests

### 7.3 Experiment Tracking Conventions (MLflow)
- Every model training run must be logged to MLflow
- Run naming convention: `{model_type}_{dataset_version}_{note}` (e.g., `xgboost_v2_tuned`)
- Required logged parameters: model type, dataset version, key hyperparameters
- Required logged metrics: MAE, RMSE, R² (train and validation)
- Required logged artifacts: serialized model, feature importance plot, confusion matrix (for LBW classification)

### 7.4 Data Versioning Conventions (DVC)
- All raw datasets tracked with DVC immediately upon receipt
- All processed/feature-engineered datasets tracked with DVC
- DVC remote configured for all team members in Week 1
- `dvc repro` must run successfully from a clean checkout

---

## 8. Documentation Plan

All documentation is stored in `docs/` and written in Markdown.

| Document | Author | Review By | Target Completion |
|----------|--------|-----------|-------------------|
| `business_understanding.md` | PM | MLE | End of Week 2 |
| `data_report.md` | DE | PM | End of Week 5 |
| `modeling_report.md` | MLE | DE | End of Week 10 |
| `deployment_guide.md` | MOE | MLE | End of Week 14 |
| `monitoring_guide.md` | MOE | PM | End of Week 17 |
| `final_report.md` | All (sections assigned) | Advisors | End of Week 19 |
| Architectural Diagrams | MOE + PM | Full team | End of Week 14 |

### 8.1 Final Report Structure (Outline)

```
1. Abstract
2. Introduction
   2.1 Healthcare Motivation
   2.2 Project Objectives
   2.3 Report Structure
3. Related Work
   3.1 Birth Weight Prediction in Literature
   3.2 MLOps and DS Process Models
4. Methodology: MLOps-Based DS Process Model
5. Business Understanding
   5.1 Problem Formulation
   5.2 Success Criteria
   5.3 Feasibility Assessment
6. Data Engineering
   6.1 Dataset Description
   6.2 Exploratory Data Analysis
   6.3 Data Quality and Preprocessing
7. Modeling
   7.1 Model Selection and Rationale
   7.2 Training and Hyperparameter Optimization
   7.3 Evaluation Results
   7.4 Model Interpretability
8. Deployment
   8.1 System Architecture
   8.2 API Design
   8.3 CI/CD Pipeline
9. Monitoring and Utilization
   9.1 Monitoring Architecture
   9.2 Drift Detection Strategy
   9.3 Maintenance and Retraining
10. Critical Reflection
    10.1 MLOps Process Model Assessment
    10.2 Technology Choices
    10.3 Data Challenges
    10.4 Team Dynamics
    10.5 Clinical and Ethical Considerations
11. Conclusion and Future Work
References
Appendices
   A. API Specification
   B. MLflow Experiment Logs
   C. Architecture Diagrams
   D. Completed Project Checklist
```

---

## 9. Definition of Done

A task or deliverable is considered **Done** when:
- [ ] Code is implemented and passes all relevant unit/integration tests
- [ ] Code has been reviewed by at least one other team member (PR approved)
- [ ] Code is merged into `develop` (or `main` for releases)
- [ ] The corresponding section in `checklist.md` is checked off
- [ ] The corresponding documentation section is drafted (even if not final)
- [ ] MLflow experiments are logged (for all model training tasks)
- [ ] DVC is updated (for all data artifact changes)

---

## 10. Project Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **ML: Mean Absolute Error (MAE)** | < 300g on test set | MLflow logged metrics |
| **ML: R² Score** | > 0.70 on test set | MLflow logged metrics |
| **Clinical: LBW Sensitivity** | > 85% | Confusion matrix on test set |
| **API: Prediction Latency** | < 500ms (p95) | Prometheus / Grafana |
| **API: Uptime** | > 99% during demo period | Prometheus / Grafana |
| **Code: Test Coverage** | > 80% | pytest-cov |
| **Reproducibility** | `dvc repro` runs clean | Manual verification |
| **Documentation Completeness** | All 7 docs complete | Checklist review |

---

*Document version: 1.0 | Created: May 2026 | STP Natality Project*
