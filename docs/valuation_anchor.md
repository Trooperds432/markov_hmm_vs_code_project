# Valuation Anchor

Valuation inputs are local CSV/parquet files containing historical, point-in-time valuation data.
The block applies `valuation_publication_lag_days` and uses backward-looking as-of joins.

Features include Graham intrinsic value, price-to-intrinsic, intrinsic-to-price, margin of safety,
TTM EPS, growth, bond yield, valuation z-score/percentile, deep-value flag, overvalued flag, and
data age.

Valuation can modestly boost confirmed Bull exposure or adjust portfolio ranking, but it does not
override Bear regimes or create standalone trades.
