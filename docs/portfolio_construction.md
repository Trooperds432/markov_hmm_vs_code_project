# Portfolio Construction

Portfolio mode runs the single-asset HMM pipeline for each configured symbol, ranks
assets by regime score, and allocates weights subject to max position count and max
weight constraints.

Supported weighting methods are equal weight, inverse volatility, signal weighted, and
a simple risk-parity approximation. Returns use lagged weights so rebalance decisions
do not use same-bar future returns.

Use:

```bash
python main.py --portfolio --config configs/portfolio_example.json --no-plots
```

Outputs include equity curve, weights, trades, stats, and latest asset signals.
