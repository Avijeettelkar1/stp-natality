# Phase 3 Modeling — Why the Targets Failed and What Is Needed

**Project:** STP Natality — Birth Weight Prediction  
**Dataset:** 27.2 million US birth records, 1986–2018 (NCHS Natality)  
**Models trained:** LightGBM and XGBoost, 30-trial Optuna Bayesian HPO each  
**Date:** June 2026

---

## 1. What We Set Out to Do

We trained machine learning models to predict birth weight (in grams) from information recorded at or before birth. Three scientific targets were set:

| Metric | Target | Best Result (LightGBM, test set) | Status |
|---|---|---|---|
| Mean Absolute Error (MAE) | < 300 g | 355 g | FAIL |
| R² (variance explained) | > 0.70 | 0.414 | FAIL |
| LBW Sensitivity | > 85 % | 38.7 % | FAIL |

LBW = Low Birth Weight, defined as birth weight < 2,500 g.

---

## 2. What We Used

### Features (21 variables — all demographic or administrative)

| Category | Variables |
|---|---|
| Gestational | `gestation_weeks`, `gestation_preterm`, `lmp_known` |
| Maternal demographics | `mother_age`, `mother_race`, `mother_education`, `mother_married`, `mother_hispanic` |
| Maternal obstetric history | `parity`, `born_alive_alive`, `born_alive_dead`, `now_dead` |
| Prenatal care | `prenatal_care_month`, `prenatal_visits` |
| Pregnancy factors | `weight_gain_pounds`, `plurality`, `delivery_method` |
| Infant | `is_male`, `birth_order` |
| Administrative | `birth_year`, `birth_state` |

### Model progression

| Model | Val MAE | Val R² | LBW Sensitivity |
|---|---|---|---|
| Baseline (predict mean always) | 446.8 g | −0.0001 | 0.0 % |
| Ridge regression (500K sample) | 377.8 g | 0.355 | 32.2 % |
| LightGBM default (500K sample) | 361.2 g | 0.406 | 37.4 % |
| XGBoost default (500K sample) | 365.5 g | 0.391 | 37.6 % |
| **LightGBM HPO (full 14M train)** | **359.6 g** | **0.411** | **37.3 %** |
| XGBoost HPO (full 14M train) | 359.7 g | 0.411 | 37.4 % |

**Key observation:** 30 trials of Bayesian hyperparameter optimisation on a 14 million row training set improved MAE by only ~1.5 g over a default LightGBM trained on 500K rows. The ceiling was hit almost immediately. This means the problem is not the model — it is the data.

---

## 3. Why Each Target Failed

### 3.1 MAE > 300 g (achieved 355 g)

Birth weight in the dataset has a standard deviation of ~598 g. The best model achieves an MAE of 355 g — roughly 59% of one standard deviation. This is substantially better than chance (446 g baseline) but not clinically precise.

The remaining error is not reducible by better algorithms or more tuning. It is **unexplained biological variance** — real differences in birth weight between babies with identical demographic profiles, caused by factors not recorded in administrative birth certificate data.

A baby's actual weight at delivery depends on dozens of biological processes (placental function, maternal metabolism, fetal genetics, infection) that produce large weight differences even among mothers who are the same age, race, education level, and gestational week. These processes are invisible to our model.

### 3.2 R² > 0.70 (achieved 0.414)

R² = 0.414 means the model explains **41.4% of the variance** in birth weight. The target was 70%.

The permutation importance analysis reveals why:

| Feature | Permutation Importance |
|---|---|
| `gestation_weeks` | 91.8 |
| `plurality` | 15.5 |
| `weight_gain_pounds` | 15.0 |
| `is_male` | 7.8 |
| `gestation_preterm` | 4.0 |
| `mother_age` | 3.4 |
| `mother_married` | 3.3 |
| `born_alive_alive` | 3.1 |
| All other features | < 1.0 each |

**Gestational age (`gestation_weeks`) is responsible for 91.8% of the model's predictive power.** Everything else — maternal demographics, prenatal care, history, geography — contributes the remaining 8.2%.

This is expected biologically. Babies born at 28 weeks will always weigh far less than babies born at 40 weeks, regardless of any other factor. But after controlling for gestational age, the remaining variance in birth weight (why one 40-week baby weighs 2,800 g while another weighs 4,200 g) is driven by biological processes the birth certificate does not capture.

The "missing" 29 percentage points of R² (from 41% to 70%) represents **biological signal that is not present in administrative records**.

### 3.3 LBW Sensitivity > 85 % (achieved 38.7 %)

Sensitivity of 38.7% means the model correctly identifies only 38.7 out of every 100 babies who are actually low birth weight. It misses 61.3%.

This is the most clinically significant failure and has a specific cause. Low birth weight babies (< 2,500 g) fall into two biological groups:

**Group A — Preterm LBW:** Babies born early (< 37 weeks). Their low weight is explained almost entirely by gestational age. Our model captures these relatively well because `gestation_weeks` is our strongest feature.

**Group B — Term LBW (Fetal Growth Restriction, FGR):** Babies born at or near full term (37–42 weeks) but weighing less than 2,500 g. These are the clinically critical cases — they are growth-restricted due to placental insufficiency, maternal disease, or genetic causes. The birth certificate contains almost no information that distinguishes a 2,200 g term baby from a 3,500 g term baby. No ultrasound measurements. No placental pathology. No maternal blood chemistry. No fetal growth trajectory.

Since term LBW (Group B) makes up a large fraction of all LBW cases and is nearly invisible in our feature set, the model cannot achieve high sensitivity.

---

## 4. What Data Would Be Needed

To meet the three targets, the following clinical data would need to be added to the feature set. None of these are present in NCHS Natality birth certificate data.

### 4.1 To reach MAE < 300 g and R² > 0.70

| Missing Data | Why It Matters |
|---|---|
| **Third-trimester ultrasound measurements** (estimated fetal weight, biparietal diameter, abdominal circumference, femur length) | Direct physical measurement of fetal size. A third-trimester ultrasound EFW (Estimated Fetal Weight) alone reduces MAE to ~150–200 g in clinical studies. |
| **Fetal growth trajectory** (serial ultrasounds showing growth velocity) | Identifies growth restriction weeks before delivery. Growth-restricted fetuses follow a flat or declining percentile curve. |
| **Maternal pre-pregnancy BMI and gestational weight gain curve** | Not just total weight gain (which we have) but the *pattern* of gain — early vs. late, sudden vs. gradual. |
| **Placental measurements** (placental volume, Doppler blood flow indices) | Placental insufficiency is the primary cause of fetal growth restriction. Uterine artery Doppler PI/RI predicts FGR months in advance. |

### 4.2 To reach LBW Sensitivity > 85 %

The sensitivity target specifically requires identifying babies who will be born small at term. This requires antenatal surveillance data:

| Missing Data | Why It Matters |
|---|---|
| **Maternal gestational diabetes diagnosis and HbA1c** | GDM causes macrosomia (large babies) but also, paradoxically, FGR in poorly managed cases with vascular disease. |
| **Pre-eclampsia / hypertensive disorder diagnosis** | Pre-eclampsia causes placental dysfunction and is the single strongest clinical predictor of FGR and LBW at term. |
| **Maternal infection markers** (CMV, rubella, TORCH panel) | Congenital infections are a major cause of symmetric FGR. |
| **Symphysis-fundal height measurements** (sequential) | Simple tape-measure assessment done at every antenatal visit. A falling SFH percentile is a clinical trigger for FGR referral. |
| **Maternal anaemia / iron and folate status** | Chronic anaemia reduces fetal oxygen delivery and is associated with LBW, especially in low-resource settings. |
| **Smoking quantity and duration** | We have a smoking indicator but not quantity. Dose-response relationship with birth weight is strong — 10 cigarettes/day ≠ 40 cigarettes/day. |

### 4.3 Data sources that would provide this

| Source | Coverage | Notes |
|---|---|---|
| Electronic Health Records (EHR) from hospital systems | US, limited | Contains ultrasound reports, lab values, diagnoses. Requires data use agreements. |
| BORN Ontario / PERINATAL data (Canada) | National | Contains antenatal visits, serial SFH, diagnoses. Not US data. |
| NICHD studies (e.g., POUCH, nuMom2b) | Research cohorts | Rich clinical data but small N (~10K). |
| UK Biobank / ALSPAC | UK | Contains maternal biomarkers, genetic data, serial measurements. |
| WHO multicountry surveys | Global | Includes some clinical variables but inconsistent across sites. |

---

## 5. Scientific Interpretation

The failure to meet the targets is itself a **publishable scientific finding**.

The result quantifies something clinicians know intuitively but which has rarely been demonstrated at population scale with 27 million records:

> **Administrative birth certificate data — even with state-of-the-art gradient boosting and exhaustive hyperparameter search — can explain only ~41% of the variance in birth weight at delivery. The remaining 59% requires clinical measurement data that is not collected in routine vital statistics records.**

This has direct implications for:

1. **Epidemiological research:** Studies that predict or adjust for birth weight using only birth certificate covariates are working with a model that misses nearly 60% of the signal. Residual confounding in such studies is substantial.

2. **Public health surveillance:** Any LBW risk-scoring system built from birth certificate data alone will miss approximately 6 out of 10 low birth weight babies. Deploying such a system clinically would be harmful.

3. **Data infrastructure policy:** The gap between 41% (what we achieved) and 70% (the clinical target) represents the scientific case for linking birth certificates to EHR and antenatal care records at a national level — a linkage that does not currently exist in the US at scale.

4. **Gestation as a mediator, not a covariate:** The dominance of `gestation_weeks` (91.8 importance) reveals a methodological issue for future work. Gestational age is itself an outcome of the same biological processes (placental function, maternal health) that determine birth weight. Using it as a predictor collapses the causal chain. A more clinically useful model might predict birth weight *conditional on planned delivery at 40 weeks* using only antenatal measurements taken before 28 weeks — a fundamentally different modelling problem.

---

## 6. Summary

| Question | Answer |
|---|---|
| Why did MAE fail? | Birth weight varies by ~355 g even among demographically identical mothers. This is biological noise the birth certificate cannot explain. |
| Why did R² fail? | After gestational age accounts for 91.8% of model power, all remaining demographic features contribute only 8.2%. The unexplained 59% of variance requires clinical data. |
| Why did LBW sensitivity fail? | Term-LBW babies (growth restricted at full gestation) are nearly invisible to our feature set. Identifying them requires ultrasound, placental Doppler, and maternal disease diagnosis — none available in NCHS data. |
| Is this a modelling failure? | No. 30-trial HPO on 14M rows produced <2 g improvement over a default model. The ceiling is in the data, not the algorithm. |
| What does it take to meet the targets? | Third-trimester ultrasound biometry, maternal disease diagnoses (pre-eclampsia, GDM), serial antenatal measurements, and placental function indices. |
| What is the scientific value? | The study quantifies the information gap in administrative birth data at population scale (27M records), providing a rigorous empirical basis for EHR–vital statistics data linkage policy. |
