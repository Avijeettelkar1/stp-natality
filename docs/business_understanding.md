# Business Understanding: Weight Analysis of Newborns
## STP Natality — MLOps-Based Data Science Project

> **Document Version:** 1.0  
> **Authors:** Project Team  
> **Advisors:** Prof. Dr. Klaus Turowski, M. Sc. Christian Haertel  
> **Status:** ✅ Complete  
> **Completed:** Phase 1, Week 2

---

## 1. Problem Statement

### 1.1 Healthcare Context

Birth weight is one of the most clinically significant indicators of newborn health. Abnormal birth weight — at both ends of the spectrum — is strongly associated with perinatal complications, developmental delays, and long-term health outcomes.

| Category | Threshold | Clinical Risk |
|----------|-----------|---------------|
| **Very Low Birth Weight (VLBW)** | < 1,500g (< 3.31 lbs) | High risk of respiratory distress, neurological issues, neonatal death |
| **Low Birth Weight (LBW)** | < 2,500g (< 5.51 lbs) | Increased risk of infections, feeding difficulties, developmental delay |
| **Normal** | 2,500g – 4,000g | Optimal range |
| **Macrosomia** | > 4,000g (> 8.82 lbs) | Risk of birth complications, shoulder dystocia, maternal injury |

Currently, birth weight can only be measured with certainty **at birth**. However, risk factors for abnormal birth weight are often present and observable much earlier — during routine prenatal examinations. A machine learning-based prediction system could allow physicians to identify at-risk pregnancies **proactively**, enabling earlier interventions.

### 1.2 Precise Problem Formulation

> **"Given a set of maternal and prenatal features available at a prenatal examination, predict the birth weight (in grams) of the newborn, so that physicians can identify pregnancies at risk of Low Birth Weight (LBW < 2,500g) or Macrosomia (> 4,000g) and intervene early."**

- **Task Type:** Supervised Regression (continuous output: grams)
- **Secondary Task:** Binary classification — LBW risk flag (< 2,500g)
- **Prediction Time:** During prenatal care (before birth)
- **Prediction Horizon:** Final birth weight at delivery

### 1.3 Why Machine Learning?

Traditional clinical screening for LBW relies on single-variable thresholds (e.g., gestational age < 37 weeks). A multi-feature ML model can capture complex interactions between maternal demographics, gestational factors, and birth history that simple rules cannot express. Historical literature has demonstrated MAE values of 250–400g for well-tuned models, making ML clinically viable for decision support.

---

## 2. Stakeholder Map

| Stakeholder | Role | Interest |
|-------------|------|----------|
| **Physicians / Midwives** | Primary end-users | Accurate, interpretable predictions to guide clinical decisions |
| **Patients (Pregnant Women)** | Subject of prediction | Privacy, fair treatment regardless of demographics |
| **Hospital Data Custodians** | Data providers | Data security, audit compliance |
| **Hospital Administrators** | System operators | System reliability, cost-effectiveness |
| **Research Advisors** | Project oversight | Scientific rigor, reproducibility, documentation quality |
| **Regulatory Bodies** | Compliance | GDPR, clinical device standards (EU MDR if deployed clinically) |

> **Important:** The system is designed as a **clinical decision support tool**, not an autonomous decision maker. The attending physician always retains final clinical authority.

---

## 3. Success Criteria

### 3.1 Machine Learning Performance Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **MAE (Mean Absolute Error)** | < 300g | Clinically acceptable: ±300g is within the precision of routine prenatal weight estimation |
| **RMSE (Root Mean Squared Error)** | Minimize | Penalises dangerous large errors more heavily than MAE |
| **R² Score** | > 0.70 | ≥70% variance explained indicates a practically useful model |
| **LBW Sensitivity (Recall)** | > 85% | Must not miss > 15% of true LBW cases — false negatives are dangerous |
| **LBW Specificity** | > 60% | Enough to avoid excessive false alarms |

### 3.2 System / Operational Criteria

| Metric | Target |
|--------|--------|
| **API Prediction Latency (p95)** | < 500ms |
| **API Uptime** | > 99% during monitoring demo period |
| **Test Coverage** | > 80% for `src/` modules |
| **Reproducibility** | `dvc repro` runs clean from fresh checkout |
| **CI/CD Pipeline** | All pushes to `main`/`develop` pass automated tests |

### 3.3 Documentation Criteria

All 7 phase-specific documents must be complete and internally consistent. Final scientific report must meet academic reporting standards for the institute.

---

## 4. Dataset Overview

The dataset is provided by the research institute and consists of **US Natality records** — birth certificate data collected by the CDC/NCHS, covering births across the United States from **1986 to 2018**.

| Property | Value |
|----------|-------|
| **Source** | US National Center for Health Statistics (NCHS) Natality Data |
| **Format** | Headerless CSV (no file extension) |
| **Files** | 5 (one per era: 1986, 1994, 2002, 2010, 2018) |
| **Total Records** | ~26,968,935 (~27 million) |
| **Columns** | 26 per record |
| **Target Variable** | `weight_pounds` (converted to `weight_grams` for modeling) |
| **Data Path** | `newborn_data/` |

### 4.1 File Breakdown

| File | Records | Size |
|------|---------|------|
| `newborn_1986` | ~3.09M | ~235 MB |
| `newborn_1994` | ~5.00M | ~398 MB |
| `newborn_2002` | ~6.19M | ~511 MB |
| `newborn_2010` | ~6.18M | ~529 MB |
| `newborn_2018` | ~6.51M | ~524 MB |
| **Total** | **~27M** | **~2.2 GB** |

See [`docs/data_documentation.md`](./data_documentation.md) for the full schema reference.

---

## 5. Key Feature Candidates

Based on domain knowledge and the schema, the following features are identified as highest-priority predictors:

### ✅ Strong Predictors (will definitely be included)

| Feature | Clinical Rationale |
|---------|-------------------|
| `gestation_weeks` | **Strongest single predictor.** Preterm births (< 37 weeks) are almost always low birth weight. |
| `plurality` | Twins/triplets are consistently ~500–1000g lighter than singletons. |
| `is_male` | Male neonates are on average ~100–150g heavier than females. |
| `weight_gain_pounds` | Maternal weight gain directly reflects fetal growth and nutritional status. |
| `mother_age` | U-shaped relationship: teen mothers and mothers ≥ 40 more likely to have LBW infants. |
| `born_alive_alive` / `parity` | Subsequent children tend to be heavier (parity effect). |

### 🟡 Moderate Value (will investigate)

| Feature | Note |
|---------|------|
| `mother_married` | Proxy for socioeconomic status; associated with healthcare access |
| `father_age` | High missing rate (~30%); modest effect |
| `month` | Seasonal birth weight variations reported in literature |
| `mother_residence_state` | Regional nutrition/healthcare differences |
| `year_fix` | Secular trends: average birth weight has shifted over decades |

### 🚫 Excluded (Data Leakage)

| Feature | Reason |
|---------|--------|
| `apgar_1min` | Measured 1 minute **after** birth |
| `apgar_5min` | Measured 5 minutes **after** birth |

---

## 6. Feasibility Assessment

### 6.1 Data Feasibility

| Question | Assessment |
|----------|------------|
| **Is the dataset large enough?** | ✅ Yes — ~27M records provides excellent statistical power for regression |
| **Are features available at prediction time?** | ✅ Yes — all retained features are observable during prenatal care (pre-birth) |
| **Is the target variable reliable?** | ✅ Yes — birth weight is a standard, objective measurement recorded on birth certificates |
| **Are there significant missing values?** | ⚠️ Partial — substance use columns are mostly absent; ~5–10% missingness on some features |
| **Is data leakage present?** | ⚠️ Yes — Apgar scores are post-birth measurements; **excluded from features** |
| **Is there temporal coverage sufficient for drift analysis?** | ✅ Yes — 32 years (1986–2018) allows meaningful drift analysis |

### 6.2 Technical Feasibility

| Factor | Assessment |
|--------|------------|
| **Team skills** | ✅ Python, scikit-learn, Docker, GitHub Actions — all within team competency |
| **Infrastructure** | ✅ Local development + Docker is sufficient; no cloud compute required |
| **Data volume** | ⚠️ 27M rows requires chunked loading or sampling for development iterations |
| **Model complexity** | ✅ Gradient boosting (XGBoost/LightGBM) is well-suited and proven for tabular regression |
| **MLflow + DVC** | ✅ Both tools are mature and suitable for the team size |

### 6.3 Development Strategy for Large Data

Because the dataset is ~2.2 GB and ~27M rows, we adopt a **two-phase development approach**:

1. **Development phase:** Work with a 100K–500K row sample (`nrows_per_file=20_000`) for fast iteration on preprocessing and modeling code.
2. **Production training:** Run the full pipeline on all 27M records for the final model, using Polars or chunked Pandas for memory efficiency.

---

## 7. Ethical & Legal Considerations

### 7.1 Data Anonymization

The US Natality dataset is a **de-identified public dataset** released by NCHS. It contains no personally identifiable information (PII) such as names, exact birth dates, or patient IDs. Individual records are anonymized by design.

- No GDPR consent issues for the historical research dataset.
- If deployed in a real clinical setting in the EU, input data from patients **would** constitute personal health data and require full GDPR + EU MDR compliance.

### 7.2 Algorithmic Fairness

The dataset spans decades and reflects historical social inequalities. Potential bias sources include:

| Bias Source | Risk | Mitigation |
|-------------|------|------------|
| **Socioeconomic proxies** | `mother_married`, `state` may encode race/income | Evaluate subgroup fairness by maternal age, year, state |
| **Historical data shifts** | Prenatal care quality improved significantly 1986→2018 | Include `year_fix` as feature; use temporal split |
| **Missing substance use data** | 1986–2002 files have no cigarette/alcohol data | Do not impute; flag missingness |
| **Underrepresented groups** | 50% sample areas (`record_weight=2`) may skew demographics | Use record weighting in statistical analyses |

### 7.3 Clinical Use Scope

> **This system is a decision support tool only.**  
> - Predictions must always be reviewed by a qualified clinician.  
> - The model should never autonomously trigger clinical actions.  
> - Physicians must be informed of the model's uncertainty and limitations.  
> - Predictions near clinical thresholds (e.g., 2,400–2,600g) should trigger specialist escalation.

---

## 8. Data Split Strategy

We use a **temporal split** based on `year_fix`:

| Split | Year Range | Approx. Size | Purpose |
|-------|-----------|--------------|---------|
| **Train** | < 2010 | ~65% of records | Model training |
| **Validation** | 2010 – 2015 | ~17% of records | Hyperparameter tuning |
| **Test** | ≥ 2016 | ~18% of records | Final, held-out evaluation |

**Rationale:** Random splitting would allow "future" records (e.g., from 2017) to leak into training, artificially inflating model performance. Temporal splitting reflects the real deployment scenario: the model is trained on historical data and predicts for future patients.

---

## 9. Project Phases Summary

| Phase | Duration | Key Output |
|-------|----------|-----------|
| **1. Business Understanding** | Weeks 1–2 | This document ✅ |
| **2. Data Engineering** | Weeks 2–5 | EDA notebook, preprocessing pipeline, data validation suite |
| **3. Modeling** | Weeks 5–9 | Trained models, MLflow experiments, best model in registry |
| **4. Evaluation** | Weeks 9–11 | Business evaluation report, Go/No-Go decision |
| **5. Deployment** | Weeks 11–15 | FastAPI, Docker, CI/CD pipeline |
| **6. Monitoring** | Weeks 15–18 | Grafana dashboards, Evidently drift detection |
| **7. Documentation** | Weeks 1–20 | Final scientific report |

---

## 10. Sign-Off

- [ ] Problem statement reviewed and approved by team
- [ ] KPIs and success criteria agreed upon
- [ ] Data access confirmed (files present in `newborn_data/`)
- [ ] Ethical considerations reviewed
- [ ] Proceed to Phase 2: Data Engineering

---

*Document version: 1.0 | Phase 1 Deliverable | STP Natality Project*
