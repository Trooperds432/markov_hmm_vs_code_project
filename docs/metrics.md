# Metrics

`stats.csv` includes the original baseline metrics plus richer risk and execution
diagnostics.

Core return metrics include strategy and buy-hold total return, CAGR, max drawdown,
Sharpe, average absolute position, trade count, win rate, profit factor, and average
trade return.

Additional metrics include annualized volatility, Sortino, Calmar, exposure-adjusted
CAGR, turnover, annualized turnover, beta/annualized alpha to the asset, optional
beta/alpha to benchmark, information ratio, VaR 95, CVaR 95, worst day, skew, and
kurtosis.

The information ratio compares strategy returns against exposure-matched passive asset
returns: `strategy_ret - asset_ret * avg_abs_position`. This makes the comparison less
punitive for strategies that intentionally hold less exposure than buy-and-hold.

`equity_curve.csv` also includes `rolling_sharpe_63`, `rolling_sharpe_252`, and
`rolling_drawdown_252`.
