# Forecast Benchmarks

Forecast benchmarks are optional. The base install always has a naive rolling forecast
baseline. If `statsforecast` is installed, the framework can use the same output
contract for AutoARIMA, AutoETS, and AutoTheta adapters.

Volatility forecasts are written to `volatility_forecasts.csv` because the output can
contain realised volatility, EWMA volatility, GARCH, and ARCH rows. EWMA volatility
uses prior returns only: the variance at time `t` is based on information through
`t-1`.

Missing optional dependencies are recorded as structured `error` values instead of
being silently skipped.
