from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import settings

_LOCK = threading.Lock()
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[a-z0-9]{6}$")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(suffix: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{suffix}"


def run_dir(run_id: str) -> Path:
    return settings.APP_RUNS_DIR / run_id


def meta_path(run_id: str) -> Path:
    return run_dir(run_id) / "meta.json"


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp, path)


def write_meta(meta: dict) -> None:
    with _LOCK:
        _atomic_write_json(meta_path(meta["run_id"]), meta)
    rebuild_registry()


def update_meta(run_id: str, **changes: Any) -> dict | None:
    with _LOCK:
        meta = _read_meta_unlocked(run_id)
        if meta is None:
            return None
        meta.update(changes)
        _atomic_write_json(meta_path(run_id), meta)
    rebuild_registry()
    return meta


def _read_meta_unlocked(run_id: str) -> dict | None:
    path = meta_path(run_id)
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None


def read_meta(run_id: str) -> dict | None:
    with _LOCK:
        return _read_meta_unlocked(run_id)


def list_meta() -> list[dict]:
    settings.ensure_dirs()
    metas: list[dict] = []
    for child in settings.APP_RUNS_DIR.iterdir():
        if not child.is_dir() or not RUN_ID_RE.match(child.name):
            continue
        meta = read_meta(child.name)
        if meta:
            metas.append(meta)
    metas.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return metas


def rebuild_registry() -> None:
    """Rebuild run_registry.json by scanning per-run meta files."""
    metas = list_meta()
    snapshot = {
        "generated_at": now_iso(),
        "count": len(metas),
        "runs": metas,
    }
    with _LOCK:
        _atomic_write_json(settings.REGISTRY_PATH, snapshot)


def recover_orphans() -> int:
    """Mark queued/running runs from a previous process as failed."""
    fixed = 0
    for meta in list_meta():
        if meta.get("status") in {"queued", "running"}:
            update_meta(
                meta["run_id"],
                status="failed",
                error="Run was interrupted by a server restart.",
                finished_at=now_iso(),
            )
            fixed += 1
    return fixed
