from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig, config_hash


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))

    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except Exception:
        return


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def get_git_sha() -> str | None:
    return _run_git(["rev-parse", "HEAD"])


def get_working_tree_dirty() -> bool | None:
    status = _run_git(["status", "--porcelain"])
    if status is None:
        return None
    return bool(status)


def make_config_hash(config: object) -> str:
    if isinstance(config, PipelineConfig):
        return config_hash(config)
    if is_dataclass(config):
        payload = asdict(config)
    else:
        payload = config
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()[:16]


def experiment_metadata(cfg: PipelineConfig) -> dict[str, Any]:
    return {
        "random_seed": int(cfg.random_seed),
        "deterministic": bool(cfg.deterministic),
        "config_hash": make_config_hash(cfg),
        "git_commit": get_git_sha(),
        "git_dirty": get_working_tree_dirty(),
        "created_at": utc_now_iso(),
        "created_at_utc": utc_now_iso(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }


def write_experiment_metadata(path: Path, cfg: PipelineConfig) -> dict[str, Any]:
    metadata = experiment_metadata(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata
