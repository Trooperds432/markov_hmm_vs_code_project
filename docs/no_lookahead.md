# No-Lookahead Rules

The Stage 2 pipeline treats timestamp safety as a hard contract.

- HMM observations must not include future returns, targets, labels, or lead-shifted data.
- Rolling features intended for decisions at time `t` use information available at or before
  `t`; predictive rolling summaries are shifted when needed.
- Lower-frequency, event, macro, fundamentals, news, options, and Fama-French data are joined
  with backward-looking point-in-time semantics.
- Missing optional feature data writes warning rows instead of being silently filled with
  future observations.
- `feature_manifest.csv` records leakage status and whether each feature was used by the HMM.

Exact per-row HMM likelihood is not claimed. The backtest writes
`hmm_window_log_likelihood` and leaves `hmm_log_likelihood` as `NaN` unless a future
implementation can compute reliable per-sample likelihood.
