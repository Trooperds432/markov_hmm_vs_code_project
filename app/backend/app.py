from __future__ import annotations

import json
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.markov_hmm_trader.config import PROFILE_OVERRIDES
from src.markov_hmm_trader.dependency_check import check_dependencies

from . import engine, registry, settings
from .jobs import JobRunner
from .loaders import list_outputs, load_series, load_stats, load_table
from .models import ConfigCreateRequest, LoginRequest, RunRequest
from .security import check_password, password_required, require_auth

DISCLAIMER = (
    "Research and educational use only. Not investment advice and not a live trading "
    "system. Paper trading is simulation-only with no broker execution. Outputs are "
    "backtested research artifacts and are not predictions or profitability claims."
)

DOWNLOAD_SUFFIXES = {".csv", ".json", ".png", ".log", ".txt"}

app = FastAPI(title="Quant Terminal API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

runner = JobRunner()


@app.on_event("startup")
def _startup() -> None:
    settings.ensure_dirs()
    registry.recover_orphans()
    registry.rebuild_registry()


# --------------------------------------------------------------------------- #
# Public endpoints (no auth gate)
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "time": registry.now_iso(),
        "app_password_required": password_required(),
    }


@app.get("/api/auth/status")
def auth_status() -> dict:
    return {"required": password_required()}


@app.post("/api/auth/login")
def auth_login(req: LoginRequest) -> dict:
    if check_password(req.password):
        return {"ok": True, "required": password_required()}
    raise HTTPException(status_code=401, detail="Invalid app password.")


@app.get("/api/meta")
def meta() -> dict:
    return {
        "disclaimer": DISCLAIMER,
        "profiles": sorted(PROFILE_OVERRIDES),
        "modes": [
            "research",
            "backtest",
            "paper",
            "portfolio",
            "compare_models",
            "nested",
            "universe",
            "diagnostics",
            "ci_smoke",
        ],
        "covariance_types": ["diag", "full", "tied", "spherical"],
        "app_password_required": password_required(),
    }


# --------------------------------------------------------------------------- #
# Protected endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/dependencies", dependencies=[Depends(require_auth)])
def dependencies() -> dict:
    df = check_dependencies()
    rows = [
        {col: (None if v != v else v) for col, v in record.items()}  # noqa: PLR0124
        for record in df.to_dict(orient="records")
    ]
    return {"dependencies": rows}


@app.get("/api/configs", dependencies=[Depends(require_auth)])
def list_configs() -> dict:
    settings.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in settings.CONFIGS_DIR.glob("*.json"))
    return {"configs": names}


@app.get("/api/configs/{name}", dependencies=[Depends(require_auth)])
def get_config(name: str) -> dict:
    try:
        path = engine.resolve_config_path(name)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    with path.open("r", encoding="utf-8") as handle:
        return {"name": name, "content": json.load(handle)}


@app.post("/api/configs/validate", dependencies=[Depends(require_auth)])
def validate_run_config(req: RunRequest) -> dict:
    try:
        cfg, warnings = engine.build_config(req)
    except engine.ConfigError as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": []}
    return {
        "valid": True,
        "errors": [],
        "warnings": warnings,
        "resolved": {
            "ticker": cfg.ticker,
            "benchmark": cfg.benchmark,
            "start": cfg.start,
            "end": cfg.end,
            "n_states": cfg.n_states,
            "covariance_type": cfg.covariance_type,
            "train_window": cfg.train_window,
            "retrain_frequency": cfg.retrain_frequency,
            "commission_bps": cfg.commission_bps,
            "slippage_bps": cfg.slippage_bps,
            "enable_nn_confirmation": cfg.enable_nn_confirmation,
        },
    }


@app.post("/api/configs", dependencies=[Depends(require_auth)])
def create_config(req: ConfigCreateRequest) -> dict:
    path = (settings.CONFIGS_DIR / req.name).resolve()
    root = settings.CONFIGS_DIR.resolve()
    if root not in path.parents:
        raise HTTPException(status_code=400, detail="invalid config path")
    settings.CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(req.content, handle, indent=2)
    return {"ok": True, "name": req.name}


@app.post("/api/runs", dependencies=[Depends(require_auth)])
def create_run(req: RunRequest) -> dict:
    try:
        cfg, warnings = engine.build_config(req)
    except engine.ConfigError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    run_id = registry.new_run_id(secrets.token_hex(3))
    directory = registry.run_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)

    config_path = engine.write_generated_config(cfg, directory)
    argv = engine.build_argv(req.mode, config_path, directory, req.plots)

    label = req.label or f"{cfg.ticker} · {req.mode}"
    meta = {
        "run_id": run_id,
        "status": "queued",
        "mode": req.mode,
        "label": label,
        "request": req.model_dump(),
        "command": argv,
        "created_at": registry.now_iso(),
        "started_at": None,
        "finished_at": None,
        "duration_sec": None,
        "return_code": None,
        "error": None,
        "metrics": None,
        "config_warnings": warnings,
        "ticker": cfg.ticker,
        "benchmark": cfg.benchmark,
    }
    registry.write_meta(meta)
    runner.submit(run_id, argv, directory)
    return meta


@app.get("/api/runs", dependencies=[Depends(require_auth)])
def list_runs() -> dict:
    return {"runs": registry.list_meta()}


def _get_meta_or_404(run_id: str) -> dict:
    if not registry.RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="invalid run id")
    meta = registry.read_meta(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="run not found")
    return meta


@app.get("/api/runs/{run_id}", dependencies=[Depends(require_auth)])
def get_run(run_id: str) -> dict:
    return _get_meta_or_404(run_id)


@app.get("/api/runs/{run_id}/status", dependencies=[Depends(require_auth)])
def run_status(run_id: str) -> dict:
    meta = _get_meta_or_404(run_id)
    return {
        "run_id": run_id,
        "status": meta.get("status"),
        "duration_sec": meta.get("duration_sec"),
        "error": meta.get("error"),
        "metrics": meta.get("metrics"),
    }


@app.get("/api/runs/{run_id}/logs", dependencies=[Depends(require_auth)])
def run_logs(run_id: str, offset: int = Query(default=0, ge=0)) -> dict:
    meta = _get_meta_or_404(run_id)
    log_path = registry.run_dir(run_id) / "job.log"
    if not log_path.is_file():
        return {"content": "", "offset": 0, "eof": meta.get("status") in registry.TERMINAL_STATUSES}
    data = log_path.read_bytes()
    chunk = data[offset:]
    return {
        "content": chunk.decode("utf-8", errors="ignore"),
        "offset": len(data),
        "eof": meta.get("status") in registry.TERMINAL_STATUSES,
    }


@app.post("/api/runs/{run_id}/cancel", dependencies=[Depends(require_auth)])
def cancel_run(run_id: str) -> dict:
    meta = _get_meta_or_404(run_id)
    if meta.get("status") in registry.TERMINAL_STATUSES:
        return {"ok": False, "reason": "run already finished"}
    registry.update_meta(run_id, status="cancelled", error="Cancelled by user.")
    runner.cancel(run_id)
    return {"ok": True}


@app.delete("/api/runs/{run_id}", dependencies=[Depends(require_auth)])
def delete_run(run_id: str) -> dict:
    meta = _get_meta_or_404(run_id)
    if meta.get("status") == "running":
        runner.cancel(run_id)
    directory = registry.run_dir(run_id)
    root = settings.APP_RUNS_DIR.resolve()
    resolved = directory.resolve()
    if root not in resolved.parents:
        raise HTTPException(status_code=400, detail="invalid run directory")
    import shutil

    shutil.rmtree(resolved, ignore_errors=True)
    registry.rebuild_registry()
    return {"ok": True}


@app.post("/api/runs/{run_id}/rerun", dependencies=[Depends(require_auth)])
def rerun(run_id: str) -> dict:
    meta = _get_meta_or_404(run_id)
    req = RunRequest(**meta.get("request", {}))
    return create_run(req)


@app.get("/api/runs/{run_id}/stats", dependencies=[Depends(require_auth)])
def run_stats(run_id: str) -> dict:
    _get_meta_or_404(run_id)
    return load_stats(registry.run_dir(run_id))


@app.get("/api/runs/{run_id}/series", dependencies=[Depends(require_auth)])
def run_series(run_id: str) -> dict:
    _get_meta_or_404(run_id)
    return load_series(registry.run_dir(run_id))


@app.get("/api/runs/{run_id}/outputs", dependencies=[Depends(require_auth)])
def run_outputs(run_id: str) -> dict:
    _get_meta_or_404(run_id)
    return {"files": list_outputs(registry.run_dir(run_id))}


@app.get("/api/runs/{run_id}/table/{name}", dependencies=[Depends(require_auth)])
def run_table(run_id: str, name: str, limit: int | None = Query(default=None, ge=1, le=20000)) -> dict:
    _get_meta_or_404(run_id)
    return load_table(registry.run_dir(run_id), name, limit)


@app.get("/api/runs/{run_id}/download/{filename}", dependencies=[Depends(require_auth)])
def download(run_id: str, filename: str) -> FileResponse:
    _get_meta_or_404(run_id)
    directory = registry.run_dir(run_id).resolve()
    target = (directory / filename).resolve()
    if directory not in target.parents:
        raise HTTPException(status_code=400, detail="path escapes run directory")
    if target.suffix.lower() not in DOWNLOAD_SUFFIXES:
        raise HTTPException(status_code=400, detail="file type not downloadable")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(target), filename=target.name)
