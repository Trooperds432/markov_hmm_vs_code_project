from __future__ import annotations

import json
import secrets

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


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.post("/login")
def login(payload: LoginRequest) -> dict[str, str]:
    if password_required() and not check_password(payload.password):
        raise HTTPException(status_code=401, detail="Invalid password.")
    token = secrets.token_urlsafe(32)
    settings.SESSIONS.add(token)
    return {"token": token}


@app.get("/health")
def health() -> dict[str, object]:
    return {"ok": True, "disclaimer": DISCLAIMER}


@app.get("/dependencies")
def dependencies(_: None = Depends(require_auth)) -> dict[str, object]:
    return {"dependencies": [item.__dict__ for item in check_dependencies()]}


@app.get("/profiles")
def profiles(_: None = Depends(require_auth)) -> dict[str, object]:
    return {"profiles": sorted(PROFILE_OVERRIDES)}


@app.get("/configs")
def configs(_: None = Depends(require_auth)) -> dict[str, object]:
    return {"configs": engine.list_configs()}


@app.post("/configs")
def create_config(
    payload: ConfigCreateRequest,
    _: None = Depends(require_auth),
) -> dict[str, object]:
    path = engine.save_config(payload.name, payload.config)
    return {"path": str(path), "configs": engine.list_configs()}


@app.get("/runs")
def runs(_: None = Depends(require_auth)) -> dict[str, object]:
    return {"runs": registry.list_meta()}


@app.post("/runs")
def start_run(payload: RunRequest, _: None = Depends(require_auth)) -> dict[str, object]:
    argv, run_id, run_directory = engine.build_command(payload)
    registry.write_meta(
        {
            "run_id": run_id,
            "status": "queued",
            "created_at": registry.now_iso(),
            "command": argv,
            "run_directory": str(run_directory),
            "config_path": payload.config_path,
            "mode": payload.mode,
            "ticker": payload.ticker,
            "profile": payload.profile,
            "disclaimer": DISCLAIMER,
        }
    )
    runner.submit(run_id, argv, run_directory)
    return {"run_id": run_id, "status": "queued"}


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str, _: None = Depends(require_auth)) -> dict[str, object]:
    if not registry.read_meta(run_id):
        raise HTTPException(status_code=404, detail="Run not found.")
    cancelled_live = runner.cancel(run_id)
    registry.update_meta(run_id, status="cancelled", cancelled_at=registry.now_iso())
    return {"run_id": run_id, "cancelled_live_process": cancelled_live}


@app.get("/runs/{run_id}")
def run_detail(run_id: str, _: None = Depends(require_auth)) -> dict[str, object]:
    meta = registry.read_meta(run_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    outputs = list_outputs(registry.run_dir(run_id))
    return {"meta": meta, "outputs": outputs}


@app.get("/runs/{run_id}/tables/{name}")
def table(
    run_id: str,
    name: str,
    limit: int = Query(default=500, ge=1, le=5000),
    _: None = Depends(require_auth),
) -> dict[str, object]:
    return load_table(registry.run_dir(run_id), name, limit=limit)


@app.get("/runs/{run_id}/stats")
def stats(run_id: str, _: None = Depends(require_auth)) -> dict[str, object]:
    return load_stats(registry.run_dir(run_id))


@app.get("/runs/{run_id}/series/{name}")
def series(run_id: str, name: str, _: None = Depends(require_auth)) -> dict[str, object]:
    return load_series(registry.run_dir(run_id), name)


@app.get("/runs/{run_id}/download/{name}")
def download(run_id: str, name: str, _: None = Depends(require_auth)) -> FileResponse:
    if any(part in {"..", ""} for part in name.split("/")):
        raise HTTPException(status_code=400, detail="Invalid file path.")
    path = registry.run_dir(run_id) / name
    if not path.is_file() or path.suffix.lower() not in DOWNLOAD_SUFFIXES:
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path)