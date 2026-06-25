# STP Natality — Project Results
**Birth Weight Prediction | NCHS Natality Dataset | June 2026**

---

## Dataset

| Property | Value |
|---|---|
| Source | US National Center for Health Statistics (NCHS) Natality Public Use Files |
| Total records | 26,968,935 (~27 million) |
| Year coverage | 1986 – 2021 |
| Files | 5 (one per era: 1986, 1994, 2002, 2010, 2018) |
| Columns per record | 26 |
| Target variable | Birth weight (grams) |

---

## Data Split

| Split | Years | Records (approx.) |
|---|---|---|
| Train | 1986 – 2009 | ~14 million |
| Validation | 2010 – 2015 | ~4 million |
| Test | 2016 – 2021 | ~4 million |

Split strategy: temporal (year-based) to prevent future data leaking into training.

---

## Target Variable Distribution

| Statistic | Value |
|---|---|
| Mean birth weight | ~3,368 g |
| Median | ~3,402 g |
| Standard deviation | ~591 g |
| VLBW (< 1,500 g) | 1.4% of records |
| LBW (< 2,500 g) | 7.3% of records |
| Normal (2,500 – 4,000 g) | 81.6% of records |
| Macrosomia (> 4,000 g) | 11.1% of records |

---

## Features Used in Modeling (21 variables)

| Category | Features |
|---|---|
| Gestational | `gestation_weeks`, `gestation_preterm`, `lmp_known` |
| Maternal demographics | `mother_age`, `mother_race`, `mother_education`, `mother_married`, `mother_hispanic` |
| Obstetric history | `parity`, `born_alive_alive`, `born_alive_dead`, `now_dead` |
| Prenatal care | `prenatal_care_month`, `prenatal_visits` |
| Pregnancy | `weight_gain_pounds`, `plurality`, `delivery_method` |
| Infant | `is_male`, `birth_order` |
| Administrative | `birth_year`, `birth_state` |

---

## Model Results

| Model | Training data | Val MAE | Val R² | LBW Sensitivity |
|---|---|---|---|---|
| Baseline (predict mean always) | — | 446.8 g | 0.000 | 0.0% |
| Ridge Regression | 500K sample | 377.8 g | 0.355 | 32.2% |
| LightGBM (default settings) | 500K sample | 361.2 g | 0.406 | 37.4% |
| XGBoost (default settings) | 500K sample | 365.5 g | 0.391 | 37.6% |
| LightGBM (30-trial Optuna HPO) | 14M full train | 359.6 g | 0.411 | 37.3% |
| XGBoost (30-trial Optuna HPO) | 14M full train | 359.7 g | 0.411 | 37.4% |

**Best model: LightGBM with Optuna HPO (30 trials, full 14M training set)**

---

## Best Model — Final Test Set Results

| Metric | Value |
|---|---|
| MAE | 355 g |
| RMSE | 496 g |
| R² | 0.414 |
| LBW Sensitivity | 38.7% |
| LBW Specificity | ~94% |

---

## Feature Importance (Permutation, Best Model)

| Rank | Feature | Permutation Importance |
|---|---|---|
| 1 | `gestation_weeks` | 91.8 |
| 2 | `plurality` | 15.5 |
| 3 | `weight_gain_pounds` | 15.0 |
| 4 | `is_male` | 7.8 |
| 5 | `gestation_preterm` | 4.0 |
| 6 | `mother_age` | 3.4 |
| 7 | `mother_married` | 3.3 |
| 8 | `born_alive_alive` | 3.1 |
| 9–21 | All remaining features | < 1.0 each |

---

## Preprocessing Steps Applied

| Step | Action |
|---|---|
| Leakage removal | Dropped `apgar_1min`, `apgar_5min` (post-birth measurements) |
| Sentinel replacement | `father_age = 99`, `gestation_weeks = 99`, `lmp ∈ {"99","9999"}` → NaN |
| Outlier capping | `gestation_weeks` clipped to [20, 44]; `mother_age` clipped to [10, 60] |
| Target filtering | Removed birth weight records outside [300, 7000] g — < 0.02% of data |
| Target conversion | `weight_pounds × 453.592` → `weight_grams` |

## Engineered Features Created

| Feature | Description |
|---|---|
| `parity` | Total prior births (live + dead) |
| `is_multiple_birth` | 1 if twins or higher-order multiple |
| `lmp_known` | 1 if last menstrual period date was recorded |
| `birth_month_sin` / `_cos` | Cyclical encoding of birth month |
| `mother_age_group` | Ordinal bins: teen / 20s / 30s / 40+ |
| `gestation_preterm` | 1 if gestational age < 37 weeks |
| `gestation_post_term` | 1 if gestational age > 41 weeks |

---

## Experiment Tracking

| Item | Detail |
|---|---|
| Tool | MLflow |
| Experiment name | `stp-natality-phase3` |
| Total runs logged | 60+ (30 HPO trials × 2 models + baselines) |
| HPO framework | Optuna (Bayesian optimisation) |
| HPO trials per model | 30 |
| Logged per run | Parameters, MAE, RMSE, R², LBW sensitivity, model artifact |

---

*STP Natality Project — Phase 3 Results | June 2026*
