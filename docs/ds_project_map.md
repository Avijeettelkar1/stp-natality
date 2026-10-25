# Data Science Project Map

This map is the output of the Business Understanding stage. It summarizes the
project on one page and links to the detailed artifacts of each stage.

## Problem and intended use

Estimate a newborn's birth weight from information known late in pregnancy, to
help notice possible low birth weight (below 2,500 g) or high birth weight
(above 4,000 g). The system is a university research prototype, not a
diagnostic tool. Details: [Business Understanding](business_understanding.md).

## Business objectives and success criteria

- A trained model beats a mean-prediction baseline on validation MAE.
- Results are reported on a temporally held-out test set.
- Data, model and code are reproducible (DVC, MLflow, Git).
- A tested API and website serve the complete prediction pipeline.
- Drift in later data can be detected.
- Limitations are documented, not hidden.

## Data Science targets

Supervised regression of birth weight in grams, with low, typical and high
bands derived from the prediction. Evaluation uses MAE, RMSE, R² and
low-birth-weight sensitivity and specificity, overall and per subgroup.

## Requirements and regulations

Functional, non-functional, data and operations requirements, as well as data
and model regulations, are listed in
[Business Understanding](business_understanding.md#requirements).

## Team and plan

Roles follow the five role categories of the process model. The team
composition and milestones are in the [Project Plan](project_plan.md).

## Main risks

- Gestation is recorded at birth, while a prenatal user only knows the current
  or expected gestation.
- Administrative records lack ultrasound and clinical measurements, which caps
  predictive performance.
- Marital status can act as a socioeconomic proxy.
- No team member has clinical training.

## Decision

Proceed with a bounded prototype. Clinical use is out of scope.

## Artifacts by stage

| Stage | Main artifacts |
|---|---|
| Business Understanding | This map, [Business Understanding](business_understanding.md), [Project Plan](project_plan.md) |
| Data Collection, Exploration, Preparation | [Data Documentation](data_documentation.md), [Data Report](data_report.md) |
| Analysis | [Modeling Report](modeling_report.md) |
| Evaluation | [Evaluation](evaluation.md) |
| Deployment | [Deployment Guide](deployment_guide.md), [Architecture](architecture.md), [Decisions](decisions/) |
| Utilization | [Monitoring Guide](monitoring_guide.md) |
