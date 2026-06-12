# Feature Quality

`feature_quality_report.csv` and `feature_drop_report.csv` audit HMM observation candidates before
model fitting.

Features are dropped from HMM observations when they are missing, all-NaN, too sparse, zero
variance, unsafe for HMM, or above an optional correlation drop threshold. Numeric retained
features are winsorized using `feature_winsorize_limits`.

Validation-only columns such as targets, labels, forward returns, hit-rate diagnostics, and
hypothesis fields are blocked from HMM observations.
