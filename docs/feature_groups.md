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
- `options`: OpenBB/mock-chain implied volatility, skew, open-interest, volume, gamma,
  and vega proxies.
- `macro`: OpenBB/mock macro data or existing proxy-symbol fallbacks.
- `fundamentals`: optional statement/metric features with graceful partial-data handling.
- `news`: lightweight headline-count and lexicon sentiment features.
- `famafrench`: factor joins and rolling factor beta proxies.
- `alt_bars`: lightweight volume/range-bar pressure features.

`qlib_alpha` is accepted as a config alias for `alpha158_lite`.

Every feature group returns the same contract:

- feature DataFrame
- metadata DataFrame
- warnings DataFrame
- produced column list
- skipped group list with reason

Main-run diagnostics:

- `feature_group_metadata.csv`
- `feature_group_warnings.csv`
- `feature_manifest.csv`

Warnings are intentional for optional groups when the active provider lacks the required
capability. This is how OpenBB-style enrichment can remain optional without silent skips.

By default, enriched feature columns are visible in diagnostics but are not HMM observations.
Set `hmm_feature_set` explicitly to opt into a custom observation set.
