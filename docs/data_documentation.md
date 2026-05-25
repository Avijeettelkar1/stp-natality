# Data Documentation: Natality Dataset
## STP Natality – Weight Analysis of Newborns

> This document describes the raw dataset provided for the project, including its source, schema, temporal coverage, and initial quality observations.

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| **Dataset Name** | US Natality Data (Historical) |
| **Format** | CSV (no file extension) |
| **Files** | 5 (one per year group: 1986, 1994, 2002, 2010, 2018) |
| **Total Records** | **26,968,935** (across all 5 files) |
| **Total Columns** | 26 |
| **Target Variable** | `weight_pounds` (birth weight of the newborn, in pounds) |
| **Task Type** | Supervised Regression |
| **Data Path** | `newborn_data/` |

### File Inventory

| File | Year Tag | Size | Row Count |
|------|----------|------|-----------|
| `newborn_data/newborn_1986` | 1986 era | ~235 MB | **3,085,784** |
| `newborn_data/newborn_1994` | 1994 era | ~398 MB | **4,997,974** |
| `newborn_data/newborn_2002` | 2002 era | ~511 MB | **6,189,575** |
| `newborn_data/newborn_2010` | 2010 era | ~529 MB | **6,183,772** |
| `newborn_data/newborn_2018` | 2018 era | ~524 MB | **6,511,830** |
| **TOTAL** | 1986–2018 | ~2.2 GB | **26,968,935** |

> ⚠️ Files contain data from various years within each era (not strictly a single year — `year_fix` column indicates the actual year). All 5 files share the same 26-column schema.

---

## 2. Schema Reference

The schema is defined in [`natality_schema.txt`](../natality_schema.txt). Below is the complete annotated schema:

### 2.1 Temporal Features

| Column | Type | Description | Availability |
|--------|------|-------------|-------------|
| `year_fix` | Integer | Four-digit year of birth (e.g., 1986) | All files |
| `month` | Integer | Month of birth (1=January, 12=December) | All files |
| `day` | Integer | Day of birth (1–31) | All files |

### 2.2 Birth Outcome Features

| Column | Type | Description | Availability |
|--------|------|-------------|-------------|
| `weight_pounds` | Float | **🎯 TARGET: Birth weight in pounds** | All files |
| `plurality` | Integer | Number of children born (1=singleton, 2=twins, etc.) | All files |
| `is_male` | Boolean | TRUE if child is male, FALSE if female | All files |

### 2.3 Apgar Health Scores (Clinical)

| Column | Type | Description | Availability |
|--------|------|-------------|-------------|
| `apgar_1min` | Integer (0–10) | Newborn health score at 1 minute after birth | **1995–2019 only** |
| `apgar_5min` | Integer (0–10) | Newborn health score at 5 minutes after birth | **1995–2019 only** |

> ⚠️ **Data Leakage Risk:** Apgar scores are measured *after* birth. They cannot be used as predictors — they are **outcomes**, not inputs. These columns must be **excluded** from the feature set.

### 2.4 Geographic Features

| Column | Type | Description | Availability |
|--------|------|-------------|-------------|
| `state` | String (2-char) | State where birth occurred (postal code) | **Unavailable after 2021** |
| `mother_residence_state` | String (2-char) | Mother's state of residence at time of birth | All files |
| `mother_birth_state` | String (2-char) | Mother's state of birth (origin) | All files |

### 2.5 Maternal Health & Demographics

| Column | Type | Description | Availability |
|--------|------|-------------|-------------|
| `mother_age` | Integer | Mother's age at time of birth | All files |
| `mother_married` | Boolean | TRUE if mother was married at time of birth | All files |
| `gestation_weeks` | Integer | Duration of pregnancy in weeks | All files |
| `lmp` | String (MMDDYYYY) | Date of last menstrual period; unknown = "99"/"9999" | All files |
| `weight_gain_pounds` | Float | Weight gained by mother during pregnancy | All files |

### 2.6 Substance Use (⚠️ Limited Availability)

| Column | Type | Description | Availability |
|--------|------|-------------|-------------|
| `cigarette_use` | Boolean | Mother smoked cigarettes during pregnancy | **2020+ only** |
| `cigarettes_per_day` | Integer | Cigarettes per day if smoker | **2020+ only** |
| `alcohol_use` | Boolean | Mother consumed alcohol during pregnancy | **2006+ only** |
| `drinks_per_week` | Integer | Drinks per week if alcohol user | **2006+ only** |

> ⚠️ **Availability Warning:** `cigarette_use` and `cigarettes_per_day` are available starting 2020, so they will be **entirely missing** in the 1986, 1994, 2002, and 2010 files. These features will require careful handling or exclusion.

### 2.7 Birth History (Parity Information)

| Column | Type | Description | Availability |
|--------|------|-------------|-------------|
| `born_alive_alive` | Integer | Prior children currently living | All files |
| `born_alive_dead` | Integer | Prior children who have since died | All files |
| `born_dead` | Integer | Prior stillbirths (miscarriages) | All files |
| `ever_born` | Integer | Total children ever born to mother (including current) | All files |

### 2.8 Paternal Information

| Column | Type | Description | Availability |
|--------|------|-------------|-------------|
| `father_age` | Integer | Father's age at time of birth | All files |

### 2.9 Sampling Metadata

| Column | Type | Description | Availability |
|--------|------|-------------|-------------|
| `record_weight` | Integer (1 or 2) | 1 = full-reporting area; 2 = 50% sample area | All files |

> ⚠️ `record_weight = 2` means the record comes from a 50% sample. This affects how the dataset should be weighted for statistical analyses.

---

## 3. Target Variable Analysis (Initial Observations)

- **Target:** `weight_pounds` (continuous float)
- **Unit conversion note:** Convert to grams for clinical interpretation: `weight_grams = weight_pounds × 453.592`
- **Clinical thresholds (in grams):**
  - Very Low Birth Weight (VLBW): < 1,500g (≈ 3.31 lbs)
  - Low Birth Weight (LBW): < 2,500g (≈ 5.51 lbs)
  - Normal: 2,500g – 4,000g (≈ 5.51 – 8.82 lbs)
  - Macrosomia: > 4,000g (≈ 8.82 lbs)
- **Expected distribution:** Roughly normal, slightly right-skewed, centered around ~7.5 lbs (≈ 3,400g)

---

## 4. Known Data Quality Issues & Considerations

### 4.1 Missing Data Patterns (Expected)

| Column | Issue | Affected Files |
|--------|-------|----------------|
| `apgar_1min`, `apgar_5min` | NULL for records outside 1995–2019 | 1986 file |
| `cigarette_use`, `cigarettes_per_day` | NULL for all records before 2020 | ALL 5 files (2020+ not included) |
| `alcohol_use`, `drinks_per_week` | NULL for records before 2006 | 1986, 1994 files |
| `state` | May be NULL for newer records | Potentially 2018 file |
| `lmp` | Unknown values encoded as "99" or "9999" | All files |
| `plurality` | Observed NULL in sample rows | All files — needs imputation strategy |

### 4.2 Encoding Issues

| Column | Issue | Handling |
|--------|-------|---------|
| `lmp` | Non-standard date encoding (MMDDYYYY); unknown as "99"/"9999" | Parse carefully; treat unknowns as missing |
| `is_male`, `mother_married`, `cigarette_use`, `alcohol_use` | Boolean encoded as string ("true"/"false") | Cast to Python bool or int |
| `father_age` | Value 99 likely means unknown | Treat 99 as missing |
| `mother_age` | Extreme values possible (e.g., < 12 or > 55) | Validate range; treat outliers |
| `gestation_weeks` | Very high values (e.g., 40+, 47 seen in sample) | Values > 44 likely erroneous; cap or remove |

### 4.3 Data Leakage Columns (MUST EXCLUDE from features)

The following columns are measured **after birth** and must never be used as model inputs:

| Column | Reason |
|--------|--------|
| `apgar_1min` | Measured 1 minute after birth |
| `apgar_5min` | Measured 5 minutes after birth |

### 4.4 Temporal Consistency

- The filename (e.g., `newborn_1986`) does not mean all records are from 1986 — `year_fix` contains the actual year and varies within files (observed: 1986–1993 in the 1986 file).
- When combining files, ensure no duplicate records across files.
- Consider **temporal train/validation/test split**: train on earlier years, test on later years (more realistic deployment scenario).

### 4.5 Sampling Bias (`record_weight`)

- Records with `record_weight = 2` are from a **50% sample area**.
- For modeling: this column may be used as a sample weight in weighted regression.
- For population-level statistics: records with `record_weight = 2` should be upweighted by 2.

---

## 5. Feature Candidates for Modeling

Based on the schema and domain knowledge, the following features are recommended as **initial candidate predictors**:

### ✅ Strong Candidates (Expected High Predictive Value)
| Feature | Clinical Rationale |
|---------|-------------------|
| `gestation_weeks` | Strongest predictor of birth weight; premature births weigh less |
| `plurality` | Twins/triplets are consistently lower weight |
| `is_male` | Boys are on average heavier than girls |
| `mother_age` | Extremes of maternal age associated with LBW |
| `weight_gain_pounds` | Maternal weight gain directly correlates with fetal growth |
| `born_alive_alive` / `ever_born` | Parity is a known predictor (subsequent children often heavier) |

### 🟡 Moderate Candidates (May Add Value)
| Feature | Note |
|---------|------|
| `mother_married` | Proxy for socioeconomic status |
| `father_age` | Paternal factors have modest effect; high missing rate |
| `month` | Possible seasonality effect |
| `mother_residence_state` | Regional differences in nutrition/healthcare |
| `year_fix` | Temporal trends in birth weight over decades |

### ⚠️ Problematic Features (Handle with Care)
| Feature | Issue |
|---------|-------|
| `lmp` | Requires significant parsing; high missing rate |
| `alcohol_use` / `drinks_per_week` | Only available in 2006+ data |
| `cigarette_use` / `cigarettes_per_day` | Only available in 2020+ data (not in any provided file) |
| `state` / `mother_birth_state` | High cardinality; needs encoding strategy |
| `record_weight` | Metadata, not a clinical predictor; use as sample weight |

### 🚫 Excluded Features (Data Leakage or Irrelevant)
| Feature | Reason |
|---------|--------|
| `apgar_1min` | Post-birth measurement |
| `apgar_5min` | Post-birth measurement |
| `day` | No clinical relevance; potential noise |

---

## 6. Suggested Data Preparation Steps

1. **Load all 5 files** and concatenate into one DataFrame with a `source_file` column
2. **Drop leakage columns**: `apgar_1min`, `apgar_5min`
3. **Parse `lmp`**: Convert to datetime; mark "99/9999" as NaN; engineer `lmp_known` flag
4. **Cast booleans**: Convert `is_male`, `mother_married` etc. from "true"/"false" strings to integers (0/1)
5. **Handle `father_age = 99`**: Replace with NaN
6. **Handle `gestation_weeks` outliers**: Cap at 44 weeks (clinical maximum for valid records)
7. **Impute missing values**: Median for numerical; mode or new category for categorical
8. **Convert target**: Add `weight_grams = weight_pounds × 453.592` column
9. **Remove target nulls**: Drop rows where `weight_pounds` is null
10. **Temporal split**: Use year-based splitting (e.g., train: pre-2010, val: 2010–2015, test: 2016+)

---

## 7. Version Control

All raw data files must be tracked using **DVC** (not Git, due to large file sizes):

```bash
# Initialize DVC
dvc init

# Track raw data
dvc add newborn_data/newborn_1986
dvc add newborn_data/newborn_1994
dvc add newborn_data/newborn_2002
dvc add newborn_data/newborn_2010
dvc add newborn_data/newborn_2018

# Commit DVC metadata
git add newborn_data/*.dvc .gitignore
git commit -m "feat: track raw natality dataset with DVC"
```

---

*Last updated: May 2026 | STP Natality Project*
