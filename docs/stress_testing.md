# Stress Testing

The framework keeps the original replay stress tests and adds separate, inspectable
stress outputs.

Replay stress reuses finished strategy positions and tests higher costs, execution
lags, bootstrapped strategy returns, and crisis windows. This is fast and useful for
execution sensitivity.

Asset bootstrap stress samples blocks of asset returns, rebuilds synthetic close paths,
and can rerun feature generation plus HMM/backtest when enabled. This is slower, so the
default config disables full rerun paths for everyday use.

Additional outputs cover commission/slippage grids, missing-data perturbations, and
named crisis windows including dotcom, global financial crisis, covid crash, rate-hike
2022, and configurable custom windows.
