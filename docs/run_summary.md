# Run Summary

Every main run writes `run_summary.json`.

It records the run id, ticker, benchmark, runtime mode, data provider, active modules, feature
counts, row counts, warning counts, and output inventory. A module is marked active only when it is
enabled and produced a non-empty active artifact.

Use this file as the first audit stop before interpreting performance outputs.
