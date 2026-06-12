# No-Lookahead Research Features

Research-public observation features are built from trailing rolling windows. At row `t`, a feature
may use data through row `t`, but not rows after `t`. Validation reports may use forward returns, but
those columns are generated after backtest decisions and are blocked from HMM observations.

The leakage guard rejects future-like observation column names, including targets, labels, forward
returns, long-short validation returns, factor hit rates, factor decay fields, and hypothesis fields.

Perturbation tests verify that changing future rows does not alter past research-public feature
values.
