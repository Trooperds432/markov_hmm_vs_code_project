from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
OUTPUTS_DIR = REPO_ROOT / "outputs"
APP_RUNS_DIR = OUTPUTS_DIR / "app_runs"
REGISTRY_PATH = APP_RUNS_DIR / "run_registry.json"

# Only a single concurrent heavy job: each subprocess loads numpy/hmmlearn/pandas
# (hundreds of MB). Additional submissions queue and surface as "queued".
MAX_WORKERS = 1

# Optional single-password gate. When unset/empty, the app is open.
APP_PASSWORD = (os.environ.get("APP_PASSWORD") or "").strip()

# Hard ceiling so a single subprocess can never run forever on Replit.
JOB_TIMEOUT_SEC = int(os.environ.get("APP_JOB_TIMEOUT_SEC") or 60 * 30)

# Max points returned for time-series chart endpoints (keeps payloads small).
MAX_SERIES_POINTS = 1200


def ensure_dirs() -> None:
    APP_RUNS_DIR.mkdir(parents=True, exist_ok=True)
