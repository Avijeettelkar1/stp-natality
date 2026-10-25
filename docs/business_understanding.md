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

## Requirements

Requirements are grouped as in the process model: functional requirements from
domain users, non-functional requirements from project management, data
requirements, and operations requirements from infrastructure and operations.

### Functional requirements

| ID | Requirement |
|---|---|
| FR1 | A user enters information known late in pregnancy through a web form or the API. |
| FR2 | The system returns an estimated birth weight in grams and a band: low (below 2,500 g), typical (2,500 to 4,000 g), or high (above 4,000 g). |
| FR3 | Every response shows the model version and a research-use note. |
| FR4 | Implausible inputs (for example, gestation outside 20 to 44 weeks) are rejected with a message per field. |
| FR5 | Other systems can integrate through a documented REST API (`/predict`, `/health`, `/model/info`). |

### Non-functional requirements

| ID | Requirement |
|---|---|
| NFR1 | Reproducibility: data and model versions are tracked with DVC, experiments with MLflow, and dependencies with `uv.lock` on Python 3.11. |
| NFR2 | Traceability: every deployed version can be traced to a Git commit and the model version it serves. |
| NFR3 | Transparency: limitations and intended use are visible in the interface and documentation. |
| NFR4 | Maintainability: code changes pass automated formatting, linting and tests before they are merged. |
| NFR5 | Security: no credentials in code or container images; HTTPS for the public service; input validation on every request. |
| NFR6 | Cost: the prototype runs on free tiers. A cold start of about one minute after inactivity is acceptable for a research prototype. |

### Data requirements

| ID | Requirement |
|---|---|
| DR1 | Source: de-identified US natality records (NCHS public-use files), 26 documented columns, births 1986 to 2021. |
| DR2 | Target: birth weight, converted to grams. |
| DR3 | Only information available before birth may be used as input. Post-birth fields (Apgar scores, day of birth) are excluded. |
| DR4 | Splits are temporal (train before 2010, validation 2010 to 2015, test from 2016) to reflect use on future births. |
| DR5 | Raw data, processed data and the trained model are versioned with DVC and are not stored in Git. |

### Operations requirements

| ID | Requirement |
|---|---|
| OR1 | Deployment happens only through the CI/CD pipeline, after all checks pass. |
| OR2 | A previous version can be restored through a rollback workflow. |
| OR3 | The live service exposes a health endpoint that reports the deployed version and whether a model is loaded. |
| OR4 | Input data drift is checked against the training reference data. |

## Data and model regulations

This section records the team's understanding of the relevant rules. It is not
legal advice and is to be confirmed with the advisors.

- **Data source and use.** The data are de-identified public-use birth records.
  They are used for university research only. No attempt is made to link
  records with other sources or to re-identify individuals.
- **Personal data in the prototype.** The web form asks for no names,
  identifiers or contact details. If prediction inputs are stored for
  monitoring, they are stored without identifiers. Health data about
  identifiable people would be special-category data under the GDPR
  (Article 9) and is out of scope for this prototype.
- **Medical use.** Software intended to support diagnosis or treatment
  decisions can fall under the EU Medical Device Regulation and would likely be
  high-risk under the EU AI Act. This prototype is not intended for clinical
  use, makes no diagnosis, and shows a research-use note with every estimate.
- **Fairness.** Performance is reported per subgroup (sex, gestation band,
  year). Marital status can act as a socioeconomic proxy and is a documented
  risk. Geographic and substance-use fields are excluded from the model.
- **Transparency.** Limitations, including the gestation-time mismatch, are
  stated in the interface, the documentation and every demonstration.

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

This is the first iteration of the project, so no existing system had to be
discontinued. The team decided to proceed with a bounded prototype. Deployment is allowed when the software is
reproducible and the selected model beats the baseline. Clinical use remains out
of scope regardless of test-set performance.
