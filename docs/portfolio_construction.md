# Portfolio Construction

Portfolio mode runs the single-asset HMM pipeline for each configured symbol, ranks
assets by regime score, and allocates weights subject to max position count and max
weight constraints.

Supported modes are `single_asset`, `top_k`, and `risk_budgeted`. The risk-budgeted
mode starts from inverse volatility, scales exposure down toward
`portfolio_target_vol`, then enforces the cash buffer and max asset weight cap.

Supported weighting methods are equal weight, inverse volatility, signal weighted, and
a simple risk-parity approximation. Returns use lagged weights so rebalance decisions
do not use same-bar future returns. The CLI `--portfolio` route preserves top-k
behavior unless the config explicitly requests `risk_budgeted`.

Use:

```bash
python main.py --portfolio --config configs/portfolio_example.json --no-plots
```

Outputs include equity curve, weights, trades, stats, and latest asset signals.
