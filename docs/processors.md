# Processor Chain

Stage 2E adds a small Qlib-style processor layer for reusable data hygiene.

Available processors:

- `DropMissingProcessor`
- `FillMissingProcessor`
- `WinsorizeProcessor`
- `ZScoreProcessor`
- `RollingZScoreProcessor`
- `FeatureColumnFilter`
- `RegimeLabelProcessor`
- `AsOfJoinProcessor`
- `FrequencyAlignmentProcessor`
- `LeakageGuardProcessor`
- `ProcessorChain`

Rules:

- Fitted processors should be fit on train windows only when used in walk-forward code.
- Rolling processors use trailing windows and shift before scoring the current row.
- As-of joins use backward-looking `merge_asof`, never forward joins.
- The leakage guard blocks obvious target/future-like feature names.

The default HMM feature path is unchanged. The processor layer is available for enriched
feature research and future walk-forward integrations.
