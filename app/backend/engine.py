from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from src.markov_hmm_trader.config import (
    apply_overrides,
    apply_profile,
    apply_quick_mode,
    load_config,
)
from src.markov_hmm_trader.config_validation import assert_config_valid, validate_config

from . import settings
from .models import RunRequest

# Engine config fields users may override from the Run panel.
_OVERRIDE_FIELDS = (
    "ticker",
    "benchmark",
    "start",
    "end",
    "n_states",
    "covariance_type",
    "commission_bps",
    "slippage_bps",
    "train_window",
    "retrain_frequency",
    "random_seed",
    "enable_diagnostics",
    "enable_stress_tests",
    "enable_nn_confirmation",
    "enable_observable_markov",
    "enable_forecast_benchmarks",
    "enable_volatility_forecasts",
)

# The ONLY CLI flags the runner is ever allowed to emit.
_ALLOWED_FLAGS = {
    "--config",
    "--output-dir",
    "--no-plots",
    "--plots",
    "--no-diagnostics",
    "--no-stress",
    "--mode",
    "--paper",
    "--portfolio",
    "--compare-models",
    "--nested",
    "--universe",
}

_MODE_FLAGS: dict[str, list[str]] = {
    "research": [],
    "backtest": [],
    "paper": ["--paper"],
    "portfolio": ["--portfolio"],
    "compare_models": ["--compare-models"],
    "nested": ["--nested"],
    "universe": ["--universe"],
    "diagnostics": ["--mode", "diagnostics"],
    "ci_smoke": ["--mode", "ci_smoke"],
}


def resolve_config_path(name: str) -> Path:
    """Resolve a config filename strictly inside configs/."""
    candidate = (settings.CONFIGS_DIR / name).resolve()
    root = settings.CONFIGS_DIR.resolve()
    if root not in candidate.parents and candidate != root:
        raise ValueError("config path escapes configs/")
    if not candidate.is_file():
        raise FileNotFoundError(f"config '{name}' not found")
    return candidate


class ConfigError(ValueError):
    pass


def build_config(req: RunRequest):
    """Build & validate a PipelineConfig from a run request. Raises ConfigError."""
    try:
        base_path = resolve_config_path(req.config)
    except (FileNotFoundError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc

    cfg = load_config(base_path)
    try:
        cfg = apply_profile(cfg, req.profile)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc

    overrides = {
        field: getattr(req, field) for field in _OVERRIDE_FIELDS if getattr(req, field) is not None
    }
    cfg = apply_overrides(cfg, **overrides)
    if req.quick:
        cfg = apply_quick_mode(cfg)

    warnings = [
        {"field": issue.field, "message": issue.message}
        for issue in validate_config(cfg)
        if issue.severity == "warning"
    ]
    try:
        assert_config_valid(cfg)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return cfg, warnings


def write_generated_config(cfg, run_dir: Path) -> Path:
    path = run_dir / "config.generated.json"
    payload = asdict(cfg)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def build_argv(mode: str, config_path: Path, run_dir: Path, plots: bool) -> list[str]:
    if mode not in _MODE_FLAGS:
        raise ConfigError(f"unsupported mode '{mode}'")
    argv = [
        sys.executable,
        "main.py",
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
        "--plots" if plots else "--no-plots",
    ]
    argv += _MODE_FLAGS[mode]
    # Defence in depth: assert every emitted flag is on the allow-list.
    for token in argv[2:]:
        if token.startswith("--") and token not in _ALLOWED_FLAGS:
            raise ConfigError(f"flag '{token}' not allowed")
    return argv
