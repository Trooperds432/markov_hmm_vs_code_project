# Quant Terminal

## Overview
A Markov/HMM + neural-network quant **research** engine (Python CLI) wrapped in a full-stack
web app. The original engine and its `main.py` CLI are unchanged; the web layer drives it as
background jobs and visualizes its CSV outputs.

> Research & educational use only. Not investment advice, no live trading. Paper trading is
> simulation-only (no broker, no real orders).

## Architecture
- **Engine (unchanged):** `main.py` + `src/markov_hmm_trader/` — the research pipeline. Configs
  live in `configs/`. Tests in `tests/`.
- **Backend:** `app/backend/` — FastAPI (port 8000) that launches the engine as allow-listed
  `main.py` subprocess jobs, tracks per-run state in `outputs/app_runs/<run_id>/meta.json`, and
  serves parsed CSV data. See `README_APP.md` for the full layout and endpoints.
- **Frontend:** `app/frontend/` — React + Vite + TS + Tailwind (port 5000), dark "quant
  terminal" UI. Proxies `/api/*` to the backend.

## Running
Two workflows start automatically:
- `Backend API` — `uvicorn app.backend.app:app --host 0.0.0.0 --port 8000` (must bind `0.0.0.0`
  for Replit port detection).
- `Start application` — `cd app/frontend && npm run dev` (Vite on port 5000, webview).

Fast offline smoke run: Backtests page → base config `quick_synthetic_provider.json` + Quick
mode (synthetic data, no network, ~8s).

## Constraints
- **Do not modify `main.py` or the engine's research logic / outputs.** The web app only wraps
  and reads it. UI run options are mapped to a generated config + an allow-list of CLI flags.
- App-only Python deps are in `requirements-app.txt`; engine deps in `requirements*.txt`.

## User preferences
- (none recorded yet)
