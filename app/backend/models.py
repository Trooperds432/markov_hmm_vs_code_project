from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.markov_hmm_trader.config import PROFILE_OVERRIDES

RunMode = Literal[
    "research",
    "backtest",
    "paper",
    "portfolio",
    "compare_models",
    "nested",
    "universe",
    "diagnostics",
    "ci_smoke",
]

_TICKER_RE = re.compile(r"^[A-Z0-9.^=\-]{1,12}$")
_CONFIG_RE = re.compile(r"^[A-Za-z0-9_.\-]+\.json$")
_COVARIANCE = {"spherical", "diag", "full", "tied"}


def _validate_symbol(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().upper()
    if not value:
        return None
    if not _TICKER_RE.match(value):
        raise ValueError("must be 1-12 chars of A-Z 0-9 . ^ = -")
    return value


def _validate_date(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:  # noqa: TRY003
        raise ValueError("must be an ISO date YYYY-MM-DD") from exc
    return value


class RunRequest(BaseModel):
    mode: RunMode = "research"
    config: str = "default.json"
    ticker: str | None = None
    benchmark: str | None = None
    start: str | None = None
    end: str | None = None
    profile: str | None = None
    quick: bool = False
    plots: bool = False
    label: str | None = None

    # Engine knobs surfaced in the Run panel (all optional overrides).
    n_states: int | None = Field(default=None, ge=2, le=8)
    covariance_type: str | None = None
    commission_bps: float | None = Field(default=None, ge=0)
    slippage_bps: float | None = Field(default=None, ge=0)
    train_window: int | None = Field(default=None, ge=30, le=5000)
    retrain_frequency: int | None = Field(default=None, ge=1, le=2000)
    random_seed: int | None = Field(default=None, ge=0, le=10_000_000)
    enable_diagnostics: bool | None = None
    enable_stress_tests: bool | None = None
    enable_nn_confirmation: bool | None = None
    enable_observable_markov: bool | None = None
    enable_forecast_benchmarks: bool | None = None
    enable_volatility_forecasts: bool | None = None

    @field_validator("ticker", "benchmark")
    @classmethod
    def _check_symbol(cls, value: str | None) -> str | None:
        return _validate_symbol(value)

    @field_validator("start", "end")
    @classmethod
    def _check_date(cls, value: str | None) -> str | None:
        return _validate_date(value)

    @field_validator("config")
    @classmethod
    def _check_config(cls, value: str) -> str:
        value = (value or "default.json").strip()
        if "/" in value or "\\" in value or not _CONFIG_RE.match(value):
            raise ValueError("config must be a plain *.json filename in configs/")
        return value

    @field_validator("profile")
    @classmethod
    def _check_profile(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if value not in PROFILE_OVERRIDES:
            raise ValueError(f"profile must be one of {sorted(PROFILE_OVERRIDES)}")
        return value

    @field_validator("covariance_type")
    @classmethod
    def _check_cov(cls, value: str | None) -> str | None:
        if value is None or value == "":
            return None
        if value not in _COVARIANCE:
            raise ValueError(f"covariance_type must be one of {sorted(_COVARIANCE)}")
        return value


class LoginRequest(BaseModel):
    password: str = ""


class ConfigCreateRequest(BaseModel):
    name: str
    content: dict

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        value = (value or "").strip()
        if not value.endswith(".json"):
            value = f"{value}.json"
        if "/" in value or "\\" in value or not _CONFIG_RE.match(value):
            raise ValueError("name must be a plain *.json filename")
        return value
