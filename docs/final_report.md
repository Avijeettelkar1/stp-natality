# Development and Operationalization of a Newborn Weight Predictor

## Abstract

This project applies an MLOps-based Data Science process model to a newborn
weight regression use case. Historical natality records are validated,
prepared, split temporally, used in traceable model experiments, served through
a tested API, and monitored for temporal input drift. On a bounded
499,259-record real-data run, LightGBM achieved a test MAE of 362.8 g and R² of
0.389. The result supports an operational prototype but not clinical use.

## 1. Introduction

Birth weight is associated with neonatal outcomes. The project evaluates
whether administrative natality fields can support a late-pregnancy
decision-support estimate while emphasizing that the prototype is not a
clinical diagnosis.

## 2. Related work

The process follows Haertel et al.'s lifecycle stages: Business Understanding;
Data Collection, Exploration and Preparation; Analysis; Evaluation; Deployment;
and Utilization. MLOps practices are informed by Kreuzberger, Kühl, and Hirschl
(2023), especially versioning, experiment tracking, testing, deployment, and
monitoring.

## 3. Business understanding

The primary output is birth weight in grams. Success means improving on a mean
baseline and delivering a reproducible end-to-end prototype. See
[`business_understanding.md`](business_understanding.md).

## 4. Data

Five era files represent the 26-column schema in
[`natality_schema.txt`](../natality_schema.txt). Apgar scores are excluded as
post-birth leakage. Cleaning handles documented sentinels and invalid ranges;
feature engineering creates parity, plurality, month, age, and gestation flags.
A temporal split separates training years before 2010, validation years
2010–2015, and test years from 2016.

## 5. Modeling and evaluation

Dummy mean, Ridge, and LightGBM candidates are compared by validation MAE and
tracked in MLflow. The selected model is evaluated once on the test set using
MAE, RMSE, R², and secondary low-birth-weight metrics.

The run sampled 100,000 rows from each era file. After validation, the temporal
splits contained 299,431 training, 75,083 validation, and 124,745 test rows.
LightGBM reduced validation MAE from the mean baseline's 436.6 g to 366.6 g
(16.0%). Test MAE was 362.8 g, RMSE 469.7 g, and R² 0.389. LBW sensitivity was
38.6% at 99.0% specificity. Critically, term-LBW sensitivity was only 1.3%;
therefore, the model is unsuitable for clinical screening.

## 6. Deployment

The selected sklearn pipeline is serialized with metadata and served by
FastAPI. Pydantic validates requests, and Docker provides a reproducible service
environment. CI formats, lints, tests, and builds the image.

## 7. Utilization

The API demonstrates model use and exposes health and model metadata endpoints.
A batch drift process compares historical training inputs with recent records.
Meaningful drift (KS statistic at least 0.10 and p below 0.01) appeared in
gestation, parental age, maternal weight gain, marital status, maternal age
group, and the deliberately separated year variable. Retraining and rollback
require human review.

## 8. Critical reflection

The process model prevents modeling results from being mistaken for the whole
project: business validity, deployment, and utilization have explicit
checkpoints. Its iterative structure is useful because data limitations can
force a return to Business Understanding.

The first implementation overcommitted to numerous tools and treated
unapproved clinical aspirations as pass/fail requirements. A smaller, connected
stack provides stronger reproducibility. The largest scientific limitation is
the data: administrative records omit ultrasound, laboratory, placental, and
growth-trajectory information. Completed gestation in historical records is
also not identical to gestational age known at an earlier examination.

## 9. Conclusion

The deliverable is an honest operational prototype and a reproducible study,
not a clinically validated system. The model improves materially over the
baseline, but poor LBW sensitivity—especially among term births—prevents a
clinical Go decision. Future work requires richer prenatal measurements and a
prospectively valid prediction time.

## References

- Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine Learning Operations
  (MLOps): Overview, Definition, and Architecture. *IEEE Access, 11*,
  31866–31879. https://doi.org/10.1109/ACCESS.2023.3262138
- Haertel, C., Staegemann, D., Pohl, M., & Turowski, K. (2025). Elevating Data
  Science Maturity: Toward a Process Model that Harnesses MLOps.
