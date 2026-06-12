# Quant Terminal — Web App

An institutional, dark "quant terminal" web UI wrapped around the existing
Markov / HMM quant research engine in this repository. A FastAPI backend drives the
**unchanged** Python engine as background jobs; a React + Vite + TypeScript + Tailwind
frontend lets you launch backtests, stream status/logs, and inspect every CSV the engine
produces (equity curves, HMM regimes, neural-network confirmation, stress tests,
portfolio, paper trading, experiments).

> **Research & educational use only.** Not investment advice and not a live trading
> system. Paper trading is **simulation-only** with no broker execution. Outputs are
> backtested research artifacts, not predictions or profitability claims.

The original CLI (`python main.py ...`) is untouched and keeps working exactly as before.

---

## Architecture

```
app/
  backend/            FastAPI app (port 8000, console workflow)
    app.py            Routes + ASGI app
    jobs.py           Background subprocess job runner + registry
    engine.py         Builds/validates config, invokes main.py via allow-listed flags
    loaders.py        Tolerant CSV loaders (stats, series, tables, outputs)
    registry.py       Per-run meta.json (atomic) + run_registry.json
    models.py         Pydantic request/response models
    security.py       Optional APP_PASSWORD gate + path-restricted downloads
    settings.py       Paths / env configuration
  frontend/           React + Vite + TS + Tailwind (port 5000, webview workflow)
    src/lib/          api client, types, hooks, auth + run contexts, formatters
    src/components/   Layout, charts (Recharts), tables, run form, command palette
    src/pages/        Dashboard, Backtests, Security, HMM Lab, NN, Portfolio,
                      Paper Trading, Stress, Experiments, Settings, Terminal
outputs/app_runs/<run_id>/   Per-run output dir (config.generated.json, job.log, CSVs)
outputs/app_runs/run_registry.json   Rebuildable run index
```

**Run model.** The backend never executes heavy work inside an HTTP request. Each run:
1. Builds a `PipelineConfig` in-process (base config → profile → overrides → quick-mode),
   validates it, and writes `config.generated.json` into the run directory.
2. Launches `main.py` as a **subprocess** with an allow-list of flags only
   (`--config`, `--output-dir`, mode flag, `--no-plots/--plots`). All engine knobs are
   baked into the generated config — never passed as ad-hoc shell arguments.
3. Streams output to `job.log`; status is tracked in an atomic per-run `meta.json`.

---

## Running on Replit

Two workflows are preconfigured and start automatically:

| Workflow         | Command                                              | Port | Type    |
| ---------------- | ---------------------------------------------------- | ---- | ------- |
| `Backend API`    | `uvicorn app.backend.app:app --host 0.0.0.0 --port 8000` | 8000 | console |
| `Start application` | `cd app/frontend && npm run dev`                  | 5000 | webview |

The Vite dev server (port 5000, host `0.0.0.0`, all hosts allowed) proxies `/api/*` to the
backend on `127.0.0.1:8000`, so the preview pane only needs port 5000.

> The backend must bind `0.0.0.0` (not `127.0.0.1`) or Replit's port probe marks the
> workflow as failed even though uvicorn is serving.

### Local / manual

```bash
# Backend
pip install -r requirements-app.txt        # fastapi, uvicorn, python-multipart
uvicorn app.backend.app:app --host 0.0.0.0 --port 8000

# Frontend
cd app/frontend && npm install && npm run dev
```

---

## Using the app

- **Dashboard** — KPI cards (Sharpe, return, max drawdown, win rate, CAGR, Sortino,
  Calmar, turnover), a Regime Detector (price with Bull/Bear/Sideways shading), state
  probabilities, equity vs buy-and-hold, and drawdown for the active run.
- **Backtests** — Full run-configuration panel (mode, config, ticker/benchmark, dates,
  profile, HMM states, covariance, commission/slippage, train/retrain windows, seed, and
  diagnostics/stress/NN/observable-Markov/forecast/volatility toggles). Validate before
  launching. Live status polling, streaming logs, downloadable artifacts, and a run
  registry with re-run / cancel / delete actions.
- **Security Analysis / HMM Regime Lab / Neural Network / Stress Tests** — Render real
  data from each run's CSVs (transition matrices, emissions, stability, NN forecasts,
  ensemble diagnostics, synthetic-sample stress, volatility forecasts, …). Missing optional
  files degrade gracefully with a clear "data unavailable" reason instead of crashing.
- **Portfolio / Paper Trading** — Launch portfolio/universe and paper (simulation-only)
  runs. Paper Trading carries a prominent "no broker, no real orders" notice.
- **Experiments** — Compare-models / nested launchers plus a raw artifact explorer for any
  CSV a run produced.
- **Settings** — Backend health, dependency doctor, engine metadata, and the disclaimer.
- **Command palette (⌘K / Ctrl+K)** — Quick navigation, run selection, and a one-click
  synthetic quick backtest.

A fast, network-free smoke run uses
`configs/quick_synthetic_provider.json` with **Quick mode** on (synthetic data, ~8s).

---

## Security

- **Optional password gate.** Set the `APP_PASSWORD` environment variable to require a
  password. All data/run endpoints are then gated; the password is sent as the
  `X-App-Password` header. With no `APP_PASSWORD` set, the app is open (suitable for local
  research).
- **Path-restricted downloads.** Download endpoints only serve files inside the approved
  `outputs/app_runs/<run_id>/` directories — no arbitrary file access.
- **No arbitrary shell.** Runs are launched with `sys.executable` and an argv list (never
  `shell=True`); only allow-listed flags are passed.

---

## Limitations

- No live or paper broker execution; paper mode is a historical/simulated session.
- No financial advice and no profitability claims.
- The quant engine's research logic and outputs are unchanged — this is a UI/runner only.
- Authentication is a single optional `APP_PASSWORD` (no user accounts or roles).
