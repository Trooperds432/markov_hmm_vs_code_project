# Portfolio

Portfolio mode keeps the single-asset path as the default. When enabled, each asset is
scored independently, then combined into target weights.

Modes:

- `single_asset`
- `top_k`
- `risk_budgeted`

Weighting methods:

- `equal`
- `score`
- `vol_inverse`
- `score_vol_inverse`

Weights respect max position count, max weight per asset, and the configured cash
buffer. Portfolio mode writes weights, trades, asset signals, equity curve, and stats.
