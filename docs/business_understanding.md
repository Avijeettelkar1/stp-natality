# Business Understanding

## Decision problem

This project investigates whether historical US natality records can support an
estimate of newborn weight during a late-pregnancy or admission-to-delivery
assessment. The estimate may help a clinician notice possible low birth weight
(below 2,500 g) or macrosomia (above 4,000 g).

The software is a university research prototype. It is not a diagnosis, does not
recommend treatment, and must not replace clinical measurements or judgement.

## Analytical formulation

- Primary task: supervised regression of birth weight in grams.
- Secondary interpretation: map the continuous prediction to low, normal, or
  macrosomia bands.
- Prediction context: late pregnancy, when gestational age and the remaining
  input fields are known.
- Data contract: only the 26 columns documented in
  [`natality_schema.txt`](../natality_schema.txt).
- Leakage: Apgar scores are measured after birth and are excluded.

The dataset records completed gestation at birth, while a prenatal user knows
current gestational age. This mismatch limits prospective clinical validity and
must be stated in every demonstration.

## Stakeholders

| Stakeholder | Need |
|---|---|
| Clinician or midwife | Transparent estimate and clear limitations |
| Pregnant patient | Privacy, fairness, and human review |
| Data custodian | Reproducible and controlled data handling |
| University advisors | Correct process application and scientific evidence |
| Project team | Maintainable code and traceable decisions |

## Success criteria

The proposal does not prescribe a numerical model threshold. Therefore, the
project uses evidence-based, attainable criteria:

1. A trained model improves validation MAE over a mean-prediction baseline.
2. MAE, RMSE, R², and low-birth-weight screening metrics are reported on a
   temporally held-out test set without changing the model afterward.
3. Data, model, and code lineage are reproducible through DVC, MLflow, and Git.
4. A tested API serves the complete preprocessing and prediction pipeline.
5. A later-year batch can be compared with the reference data for drift.
6. Limitations and failed targets are documented rather than hidden.

MAE below 300 g and high low-birth-weight sensitivity may be discussed as
clinical aspirations, but they are not project go/no-go gates unless advisors
approve them.

## Data feasibility and risks

- Approximately 27 million de-identified records are represented by five DVC
  pointers. Raw files are not stored in Git.
- Data spans multiple eras; field availability and collection practice change
  over time.
- Apgar values leak post-birth information.
- Smoking and alcohol fields are mostly unavailable in early years.
- Administrative records do not include ultrasound, placental, laboratory, or
  fetal-growth-trajectory measurements.
- Geographic and marital fields can proxy protected or socioeconomic
  characteristics and require subgroup analysis.

## Project decision

Proceed with a bounded prototype. Deployment is allowed when the software is
reproducible and the selected model beats the baseline. Clinical use remains out
of scope regardless of test-set performance.
