# Hypothesis Validation

`hypothesis_validation.csv` uses explicit statuses:
`passed`, `failed`, `inconclusive`, `insufficient_sample`, and `skipped_missing_dependency`.

Tests are not marked as passed unless a valid statistical test or clear deterministic comparison
supports the result. Forward-return and ex-post validation fields are diagnostics-only and are not
HMM inputs.
