# Data Report — EDA & Preprocessing
## STP Natality — Weight Analysis of Newborns
### Phase 2: Data Engineering Deliverable

> **Authors:** STP Natality Team  
> **Date:** June 2026  
> **Related notebooks:** [`02_data_exploration.ipynb`](../notebooks/02_data_exploration.ipynb) · [`03_data_preparation.ipynb`](../notebooks/03_data_preparation.ipynb)

---

## 1. Data Provenance

### 1.1 Dataset Overview

| Property | Value |
|----------|-------|
| **Dataset Name** | US Natality Data (Historical) |
| **Source** | US National Center for Health Statistics (NCHS) — Natality Public Use Files |
| **Format** | CSV (headerless, comma-separated, no file extension) |
| **Files** | 5 (one per year-era: 1986, 1994, 2002, 2010, 2018) |
| **Total Records** | **26,968,935** (across all 5 files) |
| **Total Columns** | 26 (shared schema across all files) |
| **Target Variable** | `weight_pounds` → converted to `weight_grams` for modeling |
| **Task Type** | Supervised Regression |
| **Data Path** | `newborn_data/` |
| **Version Control** | DVC (each file has a `.dvc` stub; content excluded from Git) |

### 1.2 File Inventory

| File | Year Era | Approx. Size | Row Count |
|------|----------|-------------|-----------|
| `newborn_data/newborn_1986` | 1986–1993 | ~235 MB | 3,085,784 |
| `newborn_data/newborn_1994` | 1994–2001 | ~398 MB | 4,997,974 |
| `newborn_data/newborn_2002` | 2002–2009 | ~511 MB | 6,189,575 |
| `newborn_data/newborn_2010` | 2010–2017 | ~529 MB | 6,183,772 |
| `newborn_data/newborn_2018` | 2018–2021 | ~524 MB | 6,511,830 |
| **TOTAL** | 1986–2021 | **~2.2 GB** | **26,968,935** |

> ⚠️ Each file contains records from a **range of years**, not a single year. The `year_fix` column indicates the exact year of birth. No duplicate records were detected across files (verified by ingestion module).

### 1.3 Collection Methodology

- Data originates from US birth certificate registrations (mandatory reporting).
- Records with `record_weight = 1` are from full-reporting areas; `record_weight = 2` are from 50% sample areas.
- Records span 1986–2021, covering approximately 35 years of US births.
- No known hospital-level or geographic exclusions.

### 1.4 Known Collection Biases

- Records from **50% sample areas** (`record_weight = 2`) are underrepresented by 50% — should be upweighted for population-level statistical analyses.
- Not all features were collected across all years (see Section 3.2 for availability gaps).
- The dataset covers **US births only** — findings may not generalise to other countries or populations.

---

## 2. Schema Reference

See [`docs/data_documentation.md`](./data_documentation.md) for the full annotated schema.

The 26-column schema is divided into:

| Group | Columns | Notes |
|-------|---------|-------|
| Temporal | `year_fix`, `month`, `day` | `day` dropped (no clinical value) |
| Birth Outcome | `weight_pounds`, `plurality`, `is_male` | `weight_pounds` is the target |
| Apgar Scores | `apgar_1min`, `apgar_5min` | **LEAKAGE — excluded** |
| Geographic | `state`, `mother_residence_state`, `mother_birth_state` | High cardinality |
| Maternal Health | `mother_age`, `mother_married`, `gestation_weeks`, `lmp`, `weight_gain_pounds` | Key predictors |
| Substance Use | `cigarette_use`, `cigarettes_per_day`, `alcohol_use`, `drinks_per_week` | Very high missing rate |
| Birth History | `born_alive_alive`, `born_alive_dead`, `born_dead`, `ever_born` | Used to compute `parity` |
| Paternal | `father_age` | High missing rate (99 = unknown) |
| Metadata | `record_weight` | Sampling weight — not a feature |

---

## 3. EDA Key Findings

### 3.1 Target Variable — `weight_grams`

| Statistic | Value |
|-----------|-------|
| Mean | ~3,368 g |
| Median | ~3,402 g |
| Std | ~591 g |
| Skewness | −0.49 (slightly left-skewed) |
| Min | 300 g (clinical lower bound) |
| Max | 7,000 g (clinical upper bound) |

**Clinical category breakdown:**

| Category | Threshold | Prevalence |
|----------|-----------|------------|
| VLBW (Very Low Birth Weight) | < 1,500 g | ~1.4% |
| LBW (Low Birth Weight) | < 2,500 g | ~7.3% |
| Normal | 2,500–4,000 g | ~81.6% |
| Macrosomia | > 4,000 g | ~11.1% |

> **Clinical observation:** The distribution is approximately normal with a slight left skew. The Q-Q plot confirms mild deviation from normality in both tails, primarily driven by the LBW and macrosomia populations. The distribution is stable across the 5 year eras — no major structural shift detected. However, mean birth weight shows a subtle **downward trend** in recent decades, consistent with published epidemiology (increasing preterm birth rates, rising multiple pregnancies via ART).

### 3.2 Missing Value Analysis

| Column | Missing % | Reason | Handling Strategy |
|--------|-----------|--------|------------------|
| `cigarette_use` | ~100% | Only available 2020+ | Excluded from feature set |
| `cigarettes_per_day` | ~100% | Only available 2020+ | Excluded from feature set |
| `alcohol_use` | ~69% | Only available 2006+ | Kept; high missingness noted |
| `drinks_per_week` | ~69% | Only available 2006+ | Kept; high missingness noted |
| `father_age` | ~25% | Value `99` = unknown | Replace 99 → NaN; will impute in modeling |
| `weight_gain_pounds` | ~8% | Not always recorded | Kept; will impute median in modeling |
| `lmp` | ~35% | Encoded as "99"/"9999" when unknown | Replace with NaN; `lmp_known` flag created |
| `gestation_weeks` | ~3% | Value `99` = unknown | Replace 99 → NaN |
| `weight_pounds` | ~0.01% | Rare null entries | Rows dropped |

**Missing value pattern:** The missing data is **not random (MAR/MNAR)** — it is highly structured by data collection year:
- Substance use columns are missing in the 1986–2018 files entirely (structural, not random).
- Father age missing rate correlates with maternal age and marital status (MAR).

### 3.3 Correlation Analysis

**Top features correlated with birth weight (`weight_grams`):**

| Feature | Pearson r | Direction | Clinical Rationale |
|---------|-----------|-----------|-------------------|
| `gestation_weeks` | +0.38 | ↑ | Strongest single predictor; premature births weigh less |
| `is_multiple_birth` | −0.24 | ↓ | Twins/triplets consistently lower weight |
| `is_male` | +0.08 | ↑ | Males are on average ~150g heavier than females |
| `plurality` | −0.22 | ↓ | Collinear with `is_multiple_birth` |
| `mother_age` | +0.03 | ↑ | Very modest effect; non-linear |
| `weight_gain_pounds` | +0.13 | ↑ | Maternal weight gain correlates with fetal growth |
| `parity` | +0.05 | ↑ | Subsequent children tend to be heavier |

> **Leakage finding (confirmed):** `apgar_1min` (r ≈ +0.15) and `apgar_5min` (r ≈ +0.14) correlate with birth weight but are post-birth measurements and have been **excluded** from the feature set.

### 3.4 Temporal Trend Analysis

| Trend | Observation |
|-------|-------------|
| Mean birth weight | Slight downward trend (~20g decrease over 35 years) |
| LBW prevalence | Modest increase (~6.5% in 1986 → ~7.5% in 2018) |
| Preterm birth rate | Increasing trend (~9% in 1990 → ~11% in 2019, per literature) |
| Distribution shift | KDE plots show stable central distribution; tails widening slightly in recent eras |

> **Conclusion:** Distributional shift is present but modest. A temporal train/validation/test split is the correct strategy to avoid leaking future trends into training.

### 3.5 Data Leakage Assessment

| Column | Leakage Risk | Decision |
|--------|-------------|----------|
| `apgar_1min` | **HIGH** — measured post-birth | ✅ Dropped in preprocessing |
| `apgar_5min` | **HIGH** — measured post-birth | ✅ Dropped in preprocessing |
| `day` | Low — day of birth not useful for prenatal prediction | ✅ Dropped in preprocessing |
| `lmp` | Low — available pre-birth but requires complex parsing | `lmp_known` flag retained; raw `lmp` excluded from features |

---

## 4. Data Preparation Decisions

### 4.1 Preprocessing Pipeline

Implemented in [`src/data/preprocess.py`](../src/data/preprocess.py). Steps applied in order:

| Step | Function | Decision & Rationale |
|------|----------|---------------------|
| **1. Drop leakage columns** | `drop_leakage_columns` | Remove `apgar_1min`, `apgar_5min` (post-birth), `day` (no predictive value) |
| **2. Cast booleans** | `cast_booleans` | Convert `"true"`/`"false"` strings to Int8 (0/1) — consistent encoding |
| **3. Handle sentinels** | `handle_missing_values` | `father_age=99`, `gestation_weeks=99`, `lmp∈{"99","9999"}` → NaN |
| **4. Cap outliers** | `cap_outliers` | `gestation_weeks`: clip to [20, 44]; `mother_age`: clip to [10, 60]; flag weight outliers with `outlier_weight` column |
| **5. Convert target** | `convert_target` | Add `weight_grams = weight_pounds × 453.592` — grams are the clinical standard |
| **6. Drop null targets** | `drop_null_targets` | Remove rows where `weight_pounds` is NaN or outside [0.66, 15.43] lbs (≈ [300, 7000] g) |

**Row removal summary (full dataset):**

| Reason | Rows Removed | % of Total |
|--------|-------------|-----------|
| Null `weight_pounds` | < 500 | < 0.01% |
| Extreme clinical outliers | < 5,000 | < 0.02% |
| **Total removed** | **< 5,500** | **< 0.02%** |

### 4.2 Feature Engineering

Implemented in [`src/data/feature_engineering.py`](../src/data/feature_engineering.py). New features created:

| Feature | Source Columns | Rationale |
|---------|---------------|-----------|
| `weight_gain_kg` | `weight_gain_pounds` | Unit conversion to SI standard |
| `parity` | `born_alive_alive + born_alive_dead + born_dead` | Total prior births — clinical predictor; subsequent children tend to be heavier |
| `is_multiple_birth` | `plurality` | Binary flag for twins/triplets — consistently lower weight |
| `lmp_known` | `lmp` | Binary data quality flag — models can learn from missingness patterns |
| `birth_month_sin` / `birth_month_cos` | `month` | Cyclical encoding avoids false ordinal distance between December and January |
| `mother_age_group` | `mother_age` | Ordinal bins (teen/20s/30s/40+) — captures non-linear age effects |
| `gestation_preterm` | `gestation_weeks` | 1 if < 37 weeks — clinically significant threshold for NICU admission |
| `gestation_post_term` | `gestation_weeks` | 1 if > 41 weeks — associated with macrosomia |

### 4.3 Train/Validation/Test Split Strategy

Implemented in [`src/data/split.py`](../src/data/split.py).

**Strategy: Temporal (year-based) split** — chosen over random split because:
1. Random split would allow future-year trends to leak into training data.
2. Deployment scenario is inherently temporal: model trained on historical records, predicting on new births.
3. Year-based split evaluates generalisation to unseen time periods (realistic).

| Split | Year Range | Approximate % | Purpose |
|-------|-----------|---------------|---------|
| **Train** | 1986–2009 (< 2010) | ~70% | Model training |
| **Validation** | 2010–2015 | ~15% | Hyperparameter tuning |
| **Test** | 2016–2021 (≥ 2016) | ~15% | Final unbiased evaluation |

**LBW stratification check:** LBW rates across splits are comparable (~7%), confirming no class imbalance introduced by the temporal split.

### 4.4 Output Files

Processed splits saved to `data/processed/` in **Parquet format**:

| File | Description |
|------|-------------|
| `train.parquet` | Training data (years < 2010) |
| `val.parquet` | Validation data (years 2010–2015) |
| `test.parquet` | Test data (years ≥ 2016) — **hold-out; do not use until Phase 3 model evaluation** |

> Parquet was chosen over CSV for: (1) ~5–10× smaller file size via columnar compression, (2) exact dtype preservation (including nullable Int8/Int16), (3) faster I/O for large DataFrames.

---

## 5. Reproducibility

### Running the Pipeline

```bash
# Full dataset (recommended for final results)
python scripts/run_pipeline.py

# Dev/test run with a sample (fast)
python scripts/run_pipeline.py --nrows 200000
```

### Notebooks

| Notebook | Purpose |
|----------|---------|
| [`notebooks/02_data_exploration.ipynb`](../notebooks/02_data_exploration.ipynb) | EDA — all visualisations and findings |
| [`notebooks/03_data_preparation.ipynb`](../notebooks/03_data_preparation.ipynb) | Pipeline execution — interactive, step-by-step |

### Unit Tests

```bash
# Unit tests (no processed data required)
pytest tests/unit/ -v

# Data validation tests (requires processed Parquet splits)
pytest tests/data_validation/ -v

# Full suite with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 6. Data Quality Summary

| Check | Status |
|-------|--------|
| No null targets in any split | ✅ |
| `weight_grams` ∈ [300, 7000] g | ✅ |
| No temporal overlap between splits | ✅ |
| No leakage columns in feature set | ✅ |
| Engineered flag columns are binary (0/1) | ✅ |
| Gestation flags are mutually exclusive | ✅ |
| `weight_grams` ≈ `weight_pounds × 453.592` | ✅ |

---

## 7. Limitations & Known Issues

1. **Substance use features** (`cigarette_use`, `cigarettes_per_day`) are absent from all 5 provided files (data starts 2020+). These known predictors of LBW are unavailable for modeling.

2. **`father_age`** has a ~25% missing rate and carries limited predictive power. It will be imputed or dropped during model feature selection (Phase 3).

3. **Geographic features** (`state`, `mother_residence_state`) have high cardinality. Target encoding or grouping by region will be needed in Phase 3.

4. **Temporal distribution shift** is present but modest. The model must be monitored for drift in Phase 6.

5. **`record_weight = 2`** records (50% sample areas) are underrepresented. For clinical prevalence estimates, these rows should be upweighted; for predictive modeling, the current equal-weight approach is standard practice.

---

*Last updated: June 2026 | STP Natality Project — Phase 2: Data Engineering*
