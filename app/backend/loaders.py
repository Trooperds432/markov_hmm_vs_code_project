from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import settings

_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _clean_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (np.floating,)):
        fv = float(value)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _coerce_number(text: str) -> Any:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    if math.isnan(value) or math.isinf(value):
        return None
    if value.is_integer() and abs(value) < 1e15:
        return value
    return value


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return None
    if df.empty:
        return df
    return df


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if limit is not None and len(df) > limit:
        df = df.tail(limit)
    df = df.replace([np.inf, -np.inf], np.nan)
    records: list[dict] = []
    for row in df.to_dict(orient="records"):
        records.append({key: _clean_scalar(val) for key, val in row.items()})
    return records


def load_table(run_directory: Path, name: str, limit: int | None = None) -> dict:
    if not _NAME_RE.match(name):
        return {"available": False, "reason": "invalid table name", "columns": [], "rows": []}
    df = _read_csv(run_directory / f"{name}.csv")
    if df is None:
        return {
            "available": False,
            "reason": "Output file not produced by this run.",
            "columns": [],
            "rows": [],
        }
    if df.empty:
        return {
            "available": False,
            "reason": "Output file is empty for this run configuration.",
            "columns": list(df.columns),
            "rows": [],
        }
    return {
        "available": True,
        "reason": "",
        "columns": list(df.columns),
        "rows": _records(df, limit),
        "total_rows": int(len(df)),
    }


def load_stats(run_directory: Path) -> dict:
    df = _read_csv(run_directory / "stats.csv")
    if df is None or df.empty or "metric" not in df.columns:
        return {"available": False, "reason": "stats.csv not available.", "stats": {}}
    stats: dict[str, Any] = {}
    for _, row in df.iterrows():
        key = str(row["metric"])
        stats[key] = _coerce_number(str(row.get("value", "")))
    return {"available": True, "reason": "", "stats": stats}


_METRIC_KEYS = (
    "strategy_sharpe",
    "strategy_total_return",
    "strategy_max_drawdown",
    "win_rate",
    "strategy_cagr",
    "strategy_sortino",
    "strategy_calmar",
    "turnover",
    "exposure_adjusted_cagr",
    "avg_abs_position",
    "buy_hold_total_return",
    "trades",
)


def load_stats_metrics(run_directory: Path) -> dict | None:
    payload = load_stats(run_directory)
    if not payload["available"]:
        return None
    stats = payload["stats"]
    metrics = {key: _clean_scalar(stats.get(key)) for key in _METRIC_KEYS}
    metrics["ticker"] = stats.get("ticker")
    metrics["benchmark"] = stats.get("benchmark")
    return metrics


def _downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = math.ceil(len(df) / max_points)
    sampled = df.iloc[::step]
    if df.index[-1] not in sampled.index:
        sampled = pd.concat([sampled, df.iloc[[-1]]])
    return sampled


def _series(df: pd.DataFrame, column: str) -> list | None:
    if column not in df.columns:
        return None
    return [_clean_scalar(v) for v in df[column].tolist()]


def load_series(run_directory: Path) -> dict:
    df = _read_csv(run_directory / "equity_curve.csv")
    if df is None or df.empty:
        return {"available": False, "reason": "equity_curve.csv not available.", "points": 0}
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = df["date"].astype(str)
    df = _downsample(df, settings.MAX_SERIES_POINTS)

    regime = None
    if "decoded_state_viterbi" in df.columns:
        regime = [
            None if (v is None or (isinstance(v, float) and math.isnan(v))) else str(v)
            for v in df["decoded_state_viterbi"].tolist()
        ]
    elif {"Bull", "Bear", "Sideways"}.issubset(df.columns):
        labels = ["Bull", "Bear", "Sideways"]
        regime = []
        for _, row in df[labels].iterrows():
            vals = {lab: row[lab] for lab in labels}
            if all(pd.isna(v) for v in vals.values()):
                regime.append(None)
            else:
                regime.append(max(vals, key=lambda k: (vals[k] if pd.notna(vals[k]) else -1)))

    drawdown = _series(df, "rolling_drawdown_252")
    if drawdown is None and "strategy_equity" in df.columns:
        equity = df["strategy_equity"].astype(float)
        running_max = equity.cummax()
        dd = (equity / running_max) - 1.0
        drawdown = [_clean_scalar(v) for v in dd.tolist()]

    return {
        "available": True,
        "reason": "",
        "points": int(len(df)),
        "dates": df["date"].tolist(),
        "price": _series(df, "Close"),
        "strategy_equity": _series(df, "strategy_equity"),
        "buy_hold_equity": _series(df, "buy_hold_equity"),
        "position": _series(df, "position"),
        "drawdown": drawdown,
        "prob_bull": _series(df, "Bull"),
        "prob_bear": _series(df, "Bear"),
        "prob_sideways": _series(df, "Sideways"),
        "confidence": _series(df, "hmm_confidence"),
        "regime": regime,
    }


def list_outputs(run_directory: Path) -> list[dict]:
    if not run_directory.is_dir():
        return []
    items: list[dict] = []
    for child in sorted(run_directory.iterdir()):
        if child.is_file():
            items.append(
                {
                    "name": child.name,
                    "size": child.stat().st_size,
                    "kind": child.suffix.lstrip(".") or "file",
                }
            )
    return items
