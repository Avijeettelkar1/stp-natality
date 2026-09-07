# MLOps Process Checklist

Checked items are implemented and evidenced in the repository. Items requiring
the raw DVC data or human/advisor action remain open.

## 1. Business Understanding

- [x] Define use case, stakeholders, intended use, and non-goals
- [x] Define regression target and secondary risk bands
- [x] Replace invented pass/fail KPIs with baseline-relative criteria
- [x] Document prediction-time mismatch, leakage, ethics, and bias risks
- [ ] Team and advisor approve the revised scope

## 2. Data Collection, Exploration, and Preparation

- [x] Define one canonical 26-column data contract
- [x] Track raw files with DVC pointers
- [x] Implement schema-validating ingestion
- [x] Remove Apgar leakage and handle documented sentinels
- [x] Implement feature engineering and temporal split
- [x] Add unit and data-validation tests
- [ ] Restore raw `newborn_*` files and rerun the pipeline
- [ ] Reconcile every EDA number against the restored files

## 3. Analysis / Modeling

- [x] Implement mean, Ridge, and LightGBM candidate workflow
- [x] Select by validation MAE
- [x] Track experiments and model artifacts in MLflow
- [x] Serialize the selected complete sklearn pipeline and metadata
- [x] Reserve the test split for one final evaluation
- [ ] Run on restored data and populate verified results
- [ ] Add subgroup results by sex, gestation band, and year

## 4. Evaluation

- [x] Define a business-aligned evaluation checkpoint
- [x] Record Conditional-Go for prototype and No-Go for clinical use
- [ ] Review real test and subgroup results
- [ ] Record team/advisor Go/No-Go decision

## 5. Deployment

- [x] Implement `/health`, `/model/info`, and `/predict`
- [x] Validate requests and return a research-use warning
- [x] Add API integration tests
- [x] Add Docker and Compose definitions
- [x] Add CI formatting, linting, tests, coverage, and image build
- [ ] Build and smoke-test with the real selected model

## 6. Utilization

- [x] Implement temporal batch-drift reporting
- [x] Document monitoring, retraining, and rollback
- [ ] Generate drift report from restored processed splits
- [ ] Conduct or record the end-to-end demonstration

## Documentation and reflection

- [x] Business, modeling, evaluation, deployment, monitoring documents
- [x] Architecture and CI diagrams
- [x] Draft final report and critical reflection
- [ ] Insert verified figures and metrics
- [ ] Team peer review
- [ ] Advisor feedback and final submission
