# Feature Groups

Feature groups are additive Stage 2E feature blocks. They do not change the default HMM
feature list by themselves, which preserves existing backtest behavior.

Implemented groups:

- `price_core`: returns, rolling means, drawdown, gaps, and benchmark correlation.
- `volatility_core`: prior-window volatility, volatility ratio, and downside volatility.
- `technical`: RSI, VWAP distance, OBV slope, trend range, Bollinger width, and momentum.
- `alpha158_lite`: lightweight Qlib-inspired rolling return, slope, correlation,
  volume, volatility, and price-position features.
- `breadth`: provider-backed market breadth context when the active provider supports it.

Optional/skipped groups:

- `macro`
- `fundamentals`
- `news`
- `options`
- `famafrench`
- `alt_bars`

Every feature group returns the same contract:

- feature DataFrame
- metadata DataFrame
- warnings DataFrame
- produced column list
- skipped group list with reason

Main-run diagnostics:

- `feature_group_metadata.csv`
- `feature_group_warnings.csv`

Warnings are intentional for optional groups when the active provider lacks the required
capability. This is how OpenBB-style enrichment can remain optional without silent skips.
