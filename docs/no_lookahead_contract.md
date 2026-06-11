# No-Lookahead Contract

The research pipeline treats time alignment as part of the model contract.

Rules:

- Features at time `t` may use only information available at or before `t`.
- Rolling statistics used as predictive features must be shifted when they summarize a
  completed prior window.
- Lower-frequency and event-based data must be joined with point-in-time/as-of semantics.
- Forward filling is allowed only after the source timestamp is known.
- Columns with future-like names such as `future`, `forward`, `fwd`, `target`, or `lead`
  are blocked by the leakage guard when used as features.

Stage 2E processors:

- `AsOfJoinProcessor` uses backward `merge_asof` joins.
- `RollingZScoreProcessor` shifts the rolling window before scoring the current row.
- `LeakageGuardProcessor` wraps the Stage 2D feature leakage guard.
- `ProcessorChain` fits processors sequentially so fitted values can be scoped to train
  data in future walk-forward integrations.

CI tests use synthetic or mocked data only. They do not call live data APIs.
