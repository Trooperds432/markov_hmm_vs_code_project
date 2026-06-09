# Markov / HMM Quant Research Framework

This project researches single-asset and optional portfolio trading rules built around
Hidden Markov Model regime detection. It is a research framework, not a live trading
system, and it does not claim profitability or production readiness.

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional neural-network experiments require:

```bash
pip install -r requirements-ml.txt
```

Optional forecast benchmarks require:

```bash
pip install -r requirements-forecast.txt
```

Check optional dependency health:

```bash
python main.py --check-deps
```

## Quick Start

Default single-symbol run:

```bash
python main.py
```

Useful faster commands:

```bash
python main.py --no-plots
python main.py --no-plots --no-diagnostics --no-stress
python main.py --quick --no-plots
```

## CLI

Core flags:

- `--config`: config JSON path, default `configs/default.json`
- `--output-dir`: output directory, default `outputs`
- `--no-plots` / `--plots`: skip or force PNG chart generation
- `--no-diagnostics`: skip diagnostics CSVs
- `--no-stress`: skip stress-test CSVs
- `--ticker`, `--benchmark`, `--start`, `--end`: override config values
- `--profile`: `balanced`, `aggressive_growth`, or `low_drawdown`
- `--quick`: fewer seeds/iterations/bootstrap paths and no plots unless `--plots`
- `--check-deps`: print availability for `hmmlearn`, `statsforecast`, `torch`, `statsmodels`, and `yfinance`

Research modes:

- `--universe`: run the configured universe list
- `--nested`: nested walk-forward optimisation
- `--portfolio`: portfolio construction mode
- `--compare-models`: compare HMM, GMM, trend, buy-hold, and cash baselines
- `--paper`: simulation-only paper trading scaffold

## Outputs

Main run outputs:

- `equity_curve.csv`: backtest rows, state probabilities, positions, returns, rolling Sharpe, rolling drawdown
- `stats.csv`: total return, CAGR, drawdown, Sharpe, Sortino, Calmar, turnover, beta/alpha, information ratio, tail risk, trade stats
- `equity_plot.png`, `regime_probabilities.png`, `position_exposure.png`: optional charts

Diagnostics:

- `window_diagnostics.csv`: walk-forward window metadata and semantic state summaries
- `seed_diagnostics.csv`: HMM seed fit status and likelihoods
- `state_summary.csv`: decoded-state and probability summaries
- `hmm_model_selection.csv`: HMM candidate validation likelihood, AIC/BIC, selection flag, errors
- `hmm_emissions.csv`: component emissions, covariance summaries, occupancy, dwell time
- `hmm_transition_matrices.csv`: selected transition matrices and semantic maps
- `hmm_state_stability.csv`: semantic map changes and transition/emission distances
- `hmm_stationary_distribution.csv`: component stationary probabilities and dwell times
- `hmm_decode_comparison.csv`: Viterbi vs MAP decoded state comparison
- `hmm_component_diagnostics.csv`: component-level return, volatility, dwell, and semantic scores
- `hmm_synthetic_sample_stress.csv`: fitted-HMM sample-path stress rows
- `factor_data_coverage.csv`: downloaded factor symbol coverage
- `feature_summary.csv`: feature distribution and missingness summary
- `nn_predictions.csv`, `ensemble_diagnostics.csv`: optional NN/ensemble diagnostics
- `volatility_forecasts.csv`: realised, EWMA, GARCH, and ARCH volatility forecasts
- `model_overfit_report.csv`: train/validation/test metric gap warnings

Stress outputs:

- `stress_tests.csv`: replay cost, lag, bootstrap, and crisis summaries
- `asset_bootstrap_stress.csv`: optional asset-return bootstrap results
- `cost_slippage_stress.csv`: commission/slippage grid
- `missing_data_stress.csv`: missing-data perturbation results
- `crisis_window_stress.csv`: named crisis-window metrics

Other modes:

- Nested: `nested_window_results.csv`, `nested_candidate_results.csv`, `nested_oos_equity_curve.csv`
- Portfolio: `portfolio_equity_curve.csv`, `portfolio_weights.csv`, `portfolio_trades.csv`, `portfolio_stats.csv`, `portfolio_asset_signals.csv`
- Model comparison: `model_comparison.csv`, `model_comparison_equity_curves.csv`, `model_state_comparison.csv`, optional `model_comparison_equity.png`
- Paper simulation: `paper_orders.csv`, `paper_positions.csv`, `paper_equity.csv`, `paper_risk_checks.csv`
- Experiments: `experiments/runs/...` config snapshots, environment info, copied CSVs, and `experiments/registry/model_registry.csv`

## Common Errors

- Missing optional NN dependency: install `requirements-ml.txt` or keep `enable_nn_confirmation=false`.
- Empty data from Yahoo Finance: check ticker, benchmark, date range, and internet access.
- Too little data after feature warmup: reduce `train_window`, use `configs/quick_test.json`, or extend the date range.
- Paper mode blocked: the model registry is not approved by default, by design.

## Interpretation Guide

Regime names are semantic labels mapped from raw HMM components. When more than three
components are fitted, multiple components can aggregate into Bull, Bear, or Sideways.
Use nested out-of-sample results and stress outputs as the primary research evidence,
not a single full-sample backtest.
