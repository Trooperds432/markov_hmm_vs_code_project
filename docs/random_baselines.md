# Random Baselines

`enable_random_baseline=true` creates random timing paths using the same return series and approximate
average exposure as the HMM strategy. The random seed is fixed by `random_baseline_seed`.

Outputs:

- `random_baseline_paths.csv`
- `random_baseline_summary.csv`

The summary reports return, Sharpe, drawdown percentiles, and the fraction of random paths beating
the HMM result. This is a diagnostic for timing edge, not proof of profitability.
