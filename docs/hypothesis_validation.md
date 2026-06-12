# Hypothesis Validation

`hypothesis_validation.csv` is an ex-post diagnostics report. It uses realised forward returns only
after the backtest has produced decisions, and those fields are never allowed into HMM observations.

The report checks whether Bull-state forward returns exceed Bear-state forward returns, whether
Bear-state volatility is higher, whether state persistence is descriptive, whether strategy Sharpe
beats a deterministic random-timing baseline, whether drawdown is less severe than buy-and-hold,
and whether returns appear heavily explained by market beta.

When `scipy` is unavailable, p-values are left missing and the descriptive metrics are still
reported with warning metadata.
