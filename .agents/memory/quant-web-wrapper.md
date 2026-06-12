---
name: Quant CLI web wrapper architecture
description: How the FastAPI app drives the existing main.py research engine without changing it
---

The web app (`app/backend/`) must never modify `main.py` or the engine. It drives runs as
**background subprocesses**:

- Build & validate a `PipelineConfig` in-process (load base config from `configs/`, then
  `apply_profile` → `apply_overrides` → `apply_quick_mode`, then `assert_config_valid`).
  `asdict(cfg) → json → load_config` round-trips safely because `load_config` filters to
  valid field names and re-tuples known tuple fields.
- Write the result to `<run_dir>/config.generated.json` and invoke the engine with an
  **allow-list of CLI flags only** (`--config`, `--output-dir`, mode flag, `--no-plots/--plots`).
  All engine knobs (n_states, covariance, commission/slippage, NN/ensemble toggles) are baked
  into the generated config, NOT passed as ad-hoc flags. This keeps the surface tiny and safe.
- Use `sys.executable`, an argv list (never `shell=True`), `cwd=REPO_ROOT`.

**Why:** Keeps the CLI untouched, avoids shell injection, and supports UI knobs the CLI has
no flags for.

**Run state:** Source of truth is per-run `outputs/app_runs/<run_id>/meta.json` (atomic
`os.replace` under a lock). `run_registry.json` is rebuilt by scanning meta files. On startup,
queued/running runs from a dead process are marked failed (orphan recovery).

**Cancellation:** With a small thread pool, a *queued* run has no live `Popen` yet, so
terminating the process is a no-op. The worker MUST re-read meta at the very start and bail
if status is already `cancelled` — otherwise it overwrites `cancelled` with `running` and runs
to completion. Cancelling a *running* run also needs a terminal-status check after the process
exits so completion doesn't clobber `cancelled`.

**Loaders:** Many engine CSVs can be missing or empty depending on config (e.g.
`hmm_model_selection.csv` is empty when `enable_hmm_model_selection=false`). Loaders must
return `{available:false, reason:...}` instead of throwing. NaN/inf must be converted to null
for valid JSON. The warm-up/training window of `equity_curve.csv` has null
`decoded_state_viterbi` (regime) — expected, not a bug.
