# Paper Trading Persistence

Paper trading is simulation-only and never connects to a live broker.

Paper mode now persists account state under the active output directory by default:
`paper_account_state.json`, `paper_trade_ledger.csv`, `paper_position_history.csv`,
`paper_cash_ledger.csv`, and `paper_daily_equity.csv`.

Set `paper_reset_state=true` to start from `paper_initial_cash`. Risk checks still run before any
simulated order, including model approval, confidence, max weight, min notional, and rebalance
threshold checks.
