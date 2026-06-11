# Data Layer

Stage 2E adds a provider abstraction while keeping `yfinance` as the default.

Providers:

- `yfinance`: default live OHLCV provider, with optional CSV cache fallback.
- `csv`: local OHLCV CSV files for offline/reproducible runs.
- `synthetic`: deterministic OHLCV and breadth snapshots for tests and smoke runs.
- `openbb`: optional adapter. Missing OpenBB creates structured warnings when optional
  enrichment is requested.

Provider selection is controlled by `data_provider`. Providers expose
`supports(capability)` so feature groups can decide whether to run or write a skip row.

Cache behavior:

- `data_cache_dir` controls where `yfinance` cache CSVs are written.
- `data_failover_to_cache=true` lets `yfinance` use cache files if a live request fails
  or returns empty data.
- `cache/` is ignored by git.

Diagnostics:

- `data_provider_metadata.csv` records provider, symbol, and observed date coverage.
- `data_provider_warnings.csv` records structured warning rows such as `used_cache`,
  `live_provider_error_used_cache`, and `missing_optional_dependency`.

Optional data extras live in `requirements-data.txt`. Base installs do not require OpenBB.
