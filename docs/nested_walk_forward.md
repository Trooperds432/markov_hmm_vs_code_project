# Nested Walk-Forward

Nested optimisation separates model selection from final reporting.

Each outer window contains a train period, validation period, and untouched test period.
Candidate configurations are evaluated on train plus validation history, selected by
the configured objective, locked, and then evaluated on the test period. The combined
test results are written to `nested_oos_equity_curve.csv`.

Use:

```bash
python main.py --nested --config configs/quick_test.json --no-plots
```

Primary outputs are `nested_window_results.csv`, `nested_candidate_results.csv`, and
`nested_oos_equity_curve.csv`.
