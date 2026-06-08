# Paper Trading Safety

Paper mode is simulation-only. It does not connect to real brokers and cannot send
real-money orders.

The paper broker tracks simulated cash, positions, fills, order status, and equity.
Risk checks run before every simulated order. Checks include max position weight,
portfolio leverage, daily loss, drawdown stop, stale data, low model confidence,
missing features, and unapproved model registry entries.

Models are never approved automatically. If `paper_model_registry_required=true`,
paper mode blocks trading unless a registry row has been manually marked approved.

Use:

```bash
python main.py --paper --config configs/quick_test.json --no-plots
```
