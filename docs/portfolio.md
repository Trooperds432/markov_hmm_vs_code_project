# Portfolio

Portfolio mode keeps the single-asset path as the default. When enabled, each asset is
scored independently, then combined into target weights.

Modes:

- `single_asset`: select only the highest-ranked asset.
- `top_k`: select up to `portfolio_max_positions`.
- `risk_budgeted`: use inverse volatility weights scaled down when estimated
  portfolio volatility exceeds `portfolio_target_vol`.

Weighting methods:

- `equal`
- `score`
- `vol_inverse`
- `score_vol_inverse`

Weights respect max position count, max weight per asset, and the configured cash
buffer. The CLI `--portfolio` path keeps top-k behavior unless the config explicitly
sets `portfolio_mode` to `risk_budgeted`. Portfolio mode writes weights, trades,
asset signals, equity curve, and stats.
