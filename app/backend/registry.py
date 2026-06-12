from __future__ import annotations

import json
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import settings

_LOCK = threading.Lock()
RUN_ID_RE = re.compile(r"^\d{8}-\d{6}-[a-z0-9]{6}$")
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_run_id(suffix: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
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
        if meta is not None:
            metas.append(meta)
    return sorted(metas, key=lambda item: item.get("created_at", ""), reverse=True)


def rebuild_registry() -> None:
    settings.ensure_dirs()
    metas = []
    for child in settings.APP_RUNS_DIR.iterdir():
        if not child.is_dir() or not RUN_ID_RE.match(child.name):
            continue
        meta = _read_meta_unlocked(child.name)
        if meta is not None:
            metas.append(meta)
    metas.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    _atomic_write_json(settings.APP_RUNS_DIR / "registry.json", {"runs": metas})
