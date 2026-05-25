# Contributing Guide — STP Natality

Thank you for contributing to this project! Please read this guide before making any changes.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Avijeettelkar1/stp-natality.git
cd stp-natality
```

### 2. Set Up Your Python Environment
```bash
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Pull the Data (via DVC)
The dataset files are **not stored in Git** — they are too large (~2.2 GB). They are versioned using DVC.

```bash
# Initialize DVC (already done — just pull)
dvc pull
```

> If DVC remote is not yet configured, ask the team lead for access credentials to the shared remote storage.

### 4. Start the MLflow Tracking Server
```bash
mlflow ui --port 5000
# Open http://localhost:5000 in your browser
```

### 5. Run the Tests
```bash
pytest tests/ --cov=src
```

---

## 🌿 Branching Strategy (GitFlow)

| Branch | Purpose |
|--------|---------|
| `main` | Stable, production-ready releases only. Never commit directly. |
| `develop` | Integration branch. All features merged here first. |
| `feature/<name>` | One branch per feature or task. Branch from `develop`. |
| `hotfix/<name>` | Urgent fix to `main`. |
| `release/<version>` | Final stabilization before tagging a release. |

### Creating a Feature Branch
```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Merging a Feature
- Open a **Pull Request (PR)** from `feature/...` → `develop`
- PR must be reviewed and approved by **at least 1 other team member**
- All CI checks must pass before merging
- Delete the feature branch after merging

---

## 📝 Commit Message Convention

Follow the [Conventional Commits](https://www.conventionalcommits.org/) standard:

```
<type>(<scope>): <short description>
```

### Types
| Type | When to Use |
|------|------------|
| `feat` | New feature or notebook |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `data` | Data pipeline or dataset changes |
| `model` | Model training, HPO, evaluation |
| `deploy` | Deployment, Docker, CI/CD |
| `test` | Adding or fixing tests |
| `refactor` | Code refactoring without behavior change |
| `chore` | Maintenance (deps updates, config) |

### Examples
```
feat(data): add preprocessing pipeline for natality dataset
model(xgboost): add hyperparameter tuning with Optuna
docs(eda): complete data exploration notebook
fix(api): handle missing gestation_weeks in prediction request
deploy(docker): add docker-compose for monitoring stack
```

---

## 🧹 Code Quality Standards

All code must pass before submitting a PR:

```bash
# Format code
black src/ tests/

# Check imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

> **Coverage target:** ≥ 80% for all `src/` modules.

### Docstrings
All functions and classes must have **Google-style docstrings**:

```python
def predict_weight(features: dict) -> float:
    """Predict the birth weight of a newborn.

    Args:
        features: Dictionary of input features matching the API schema.

    Returns:
        Predicted birth weight in pounds.

    Raises:
        ValueError: If required features are missing.
    """
```

---

## 📊 MLflow Experiment Tracking

Every model training run **must** be logged to MLflow:

```python
import mlflow

with mlflow.start_run(run_name="xgboost_v1_baseline"):
    mlflow.log_params({"n_estimators": 500, "max_depth": 6})
    mlflow.log_metrics({"mae": 0.45, "rmse": 0.62, "r2": 0.71})
    mlflow.sklearn.log_model(model, "model")
```

**Naming convention:** `{model_type}_{dataset_version}_{note}`  
Example: `xgboost_v2_tuned`, `lightgbm_v1_baseline`

---

## 📁 DVC Data Versioning

When adding or updating data files:

```bash
dvc add data/raw/your_new_file.csv
git add data/raw/your_new_file.csv.dvc .gitignore
git commit -m "data: add processed feature set v2"
dvc push
```

**Never** `git add` the actual large data files — only the `.dvc` pointer files.

---

## 🔄 Pull Request Checklist

Before submitting a PR, confirm:

- [ ] Code formatted with Black and isort
- [ ] Flake8 passes with no errors
- [ ] All new code has docstrings
- [ ] Tests added/updated for new functionality
- [ ] All tests pass locally: `pytest tests/ --cov=src`
- [ ] MLflow experiments logged (for model changes)
- [ ] DVC updated and pushed (for data changes)
- [ ] Checklist.md updated if a phase task was completed
- [ ] PR description clearly explains what changed and why

---

## 👥 Team Contacts

| Role | Name | Responsibility |
|------|------|----------------|
| Project Manager | TBD | Planning, coordination, documentation |
| Data Engineer | TBD | Data pipelines, EDA |
| ML Engineer | TBD | Modeling, experiment tracking |
| MLOps Engineer | TBD | Deployment, CI/CD, monitoring |

---

*Questions? Open a GitHub Issue or reach out on the team communication channel.*
