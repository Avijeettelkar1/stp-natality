# Business Evaluation

## Decision rule

The project may proceed to prototype deployment when:

- the candidate model improves validation MAE over the mean baseline;
- the held-out test metrics and subgroup results are recorded;
- the API serves the exact training feature contract;
- known limitations are visible to users.

Failure to meet an aspirational clinical metric is a finding, not a reason to
hide or repeatedly query the test set.

## Required evidence

After real-data training, record:

1. baseline and selected-model validation MAE;
2. test MAE, RMSE, R², LBW sensitivity, and specificity;
3. errors by infant sex, gestational-age band, and year;
4. performance on term low-birth-weight cases;
5. model version and data-contract version.

## Current decision

**Conditional Go for software demonstration; No-Go for clinical use.**

The end-to-end pipeline, deployment, and monitoring can be demonstrated while
clearly labeling final metrics as pending. Clinical use is out of scope because
the dataset lacks ultrasound, laboratory, placental, and longitudinal fetal
growth measurements and has a prediction-time mismatch for gestational age.
