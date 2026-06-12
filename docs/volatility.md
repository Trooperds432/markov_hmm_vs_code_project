# Volatility Forecasts

`volatility_forecasts.csv` contains realised, EWMA, and optional ARCH/GARCH-style
volatility forecasts.

Core columns:

- `forecast_sigma`: annualized volatility forecast.
- `forecast_variance`: forecast horizon variance.
- `realised_future_sigma`: future realised volatility over the same horizon, used only for
  research diagnostics.
- `abs_error`: absolute forecast error against future realised volatility.
- `vol_rank`: rolling percentile rank of forecast volatility.
- `vol_risk_multiplier`: bounded risk multiplier.
- `error`: structured optional dependency or model warning.

EWMA uses prior returns only. ARCH/GARCH adapters remain optional and fall back with
structured errors if dependencies are unavailable.

`model_overfit_report.csv` uses realised forecast errors when present. If older forecast
outputs lack realised errors, it falls back to the previous mean-sigma proxy for backward
compatibility.
