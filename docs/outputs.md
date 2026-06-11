# Outputs

Core outputs:

- `equity_curve.csv`
- `stats.csv`
- `window_diagnostics.csv`
- `seed_diagnostics.csv`
- `state_summary.csv`

`equity_curve.csv` preserves `hidden_state` and adds HMM decode compatibility columns
`decoded_state_viterbi`, `decoded_state_map`, and `state_decode_disagreement`.

HMM outputs:

- `hmm_model_selection.csv`
- `hmm_emissions.csv`
- `hmm_transition_matrices.csv`
- `hmm_stationary_distribution.csv`
- `hmm_decode_comparison.csv`
- `hmm_component_diagnostics.csv`
- `hmm_synthetic_sample_stress.csv`
- `hmm_synthetic_sample_stress_summary.csv`

`hmm_stationary_distribution.csv` includes `state`, `probability`, `model_type`,
`n_states`, `covariance_type`, `seed`, and `n_mix` alongside component diagnostics.

Benchmark outputs:

- `observable_markov_baseline.csv`
- `observable_markov_transition_matrix.csv`
- `observable_markov_stationary_distribution.csv`
- `observable_markov_forecasts.csv`
- `observable_markov_equity_curve.csv`
- `forecast_benchmarks.csv`
- `forecast_predictions.csv`
- `forecast_model_comparison.csv`
- `forecast_confirmation_diagnostics.csv`
- `volatility_forecasts.csv`

Risk and overfit outputs:

- `stress_tests.csv`
- `asset_bootstrap_stress.csv`
- `cost_slippage_stress.csv`
- `missing_data_stress.csv`
- `crisis_window_stress.csv`
- `model_overfit_report.csv`

`model_overfit_report.csv` can contain rows for HMM, forecast, volatility, NN, and
nested optimiser candidates. Higher-is-better metrics such as likelihoods and
lower-is-better metrics such as losses are gap-adjusted before warning thresholds are
applied. Missing train/validation/test values are written as `missing_metric` rows.

NN and ensemble outputs:

- `nn_predictions.csv`
- `nn_forecasts.csv`
- `nn_oos_split_info.csv`
- `nn_training_metrics.csv`
- `ensemble_predictions.csv`
- `ensemble_diagnostics.csv`

`ensemble_mode="multi_signal"` combines the HMM score, adjusted NN probability,
observable Markov signal, forecast signal, and volatility risk multiplier when those
columns are available. Missing optional signals default to neutral values.

Portfolio and paper outputs:

- `portfolio_weights.csv`
- `portfolio_asset_signals.csv`
- `portfolio_equity_curve.csv`
- `paper_orders.csv`
- `paper_fills.csv`
- `paper_positions.csv`
- `paper_daily_equity.csv`
- `paper_equity.csv`
