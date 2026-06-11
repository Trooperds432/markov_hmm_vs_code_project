# Forecast Benchmarks

Forecast benchmarks are optional. The base install always has a naive rolling forecast
baseline. If `statsforecast` is installed, the framework can use the same output
contract for AutoARIMA, AutoETS, and AutoTheta adapters.

`forecast_benchmarks.csv` is a compatibility summary alias for
`forecast_model_comparison.csv`. `forecast_predictions.csv` stores per-date forecast
means, uncertainty bands, direction signals, split labels, and structured errors.

Volatility forecasts are written to `volatility_forecasts.csv` because the output can
contain realised volatility, EWMA volatility, GARCH, and ARCH rows. EWMA volatility
uses prior returns only: the variance at time `t` is based on information through
`t-1`. The volatility risk multiplier is clipped to the configured volatility
multiplier bounds, including the main `min_vol_mult` and `max_vol_mult` limits. The
Stage 2A `vol_rank_window` alias is accepted for the existing volatility rank
lookback.

Missing optional dependencies are recorded as structured `error` values instead of
being silently skipped.
