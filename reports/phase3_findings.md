# Phase 3 Findings

The clean bounded run used only fields from the provided 26-column schema.
LightGBM improved validation MAE by 16.0% over the mean baseline and achieved
362.8 g MAE and R² 0.389 on the held-out test split.

The overall LBW sensitivity of 38.6% is too low for clinical screening. The
subgroup split explains the practical danger: sensitivity was 59.3% for preterm
births but only 1.3% for term births. The model identifies some low weights
associated with prematurity but almost entirely misses low-weight term births.

This supports a Conditional-Go for an educational software demonstration and a
No-Go for clinical use. The evidence does not prove that the remaining error is
irreducible; it shows only that this model and administrative feature set are
insufficient. Ultrasound, longitudinal growth, laboratory, and maternal disease
measurements are reasonable candidates for future investigation.
