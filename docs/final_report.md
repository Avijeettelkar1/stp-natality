# Development and Operationalization of a Newborn Weight Predictor

## Abstract

This project applies an MLOps-based Data Science process model to a newborn
weight regression use case. Historical natality records are validated,
prepared, split temporally, used in traceable model experiments, served through
a tested API, and monitored for temporal input drift. Final model numbers remain
pending until the DVC-tracked raw files are restored.

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
MAE, RMSE, R², and secondary low-birth-weight metrics. Insert verified values
from `reports/model_metrics.json` after running the real data.

## 6. Deployment

The selected sklearn pipeline is serialized with metadata and served by
FastAPI. Pydantic validates requests, and Docker provides a reproducible service
environment. CI formats, lints, tests, and builds the image.

## 7. Utilization

The API demonstrates model use and exposes health and model metadata endpoints.
A batch drift process compares historical training inputs with recent records.
Retraining and rollback require human review.

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
not a clinically validated system. Final conclusions about predictive
performance must be added only after the tracked data is restored and the
workflow is rerun.

## References

- Kreuzberger, D., Kühl, N., & Hirschl, S. (2023). Machine Learning Operations
  (MLOps): Overview, Definition, and Architecture. *IEEE Access, 11*,
  31866–31879. https://doi.org/10.1109/ACCESS.2023.3262138
- Haertel, C., Staegemann, D., Pohl, M., & Turowski, K. (2025). Elevating Data
  Science Maturity: Toward a Process Model that Harnesses MLOps.
