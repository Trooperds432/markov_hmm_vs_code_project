from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import PipelineConfig


def resolve_paper_path(configured: str, output_dir: Path, default_name: str) -> Path:
    path = Path(configured or default_name)
    if path.is_absolute():
        return path
    if len(path.parts) > 1 and path.parts[0] == "outputs":
        return path
    return output_dir / path


def default_state(initial_cash: float) -> dict[str, Any]:
    return {
        "cash": float(initial_cash),
        "positions": {},
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "equity": float(initial_cash),
        "last_update": "",
    }


def load_paper_state(cfg: PipelineConfig, output_dir: Path) -> tuple[dict[str, Any], Path]:
    path = resolve_paper_path(cfg.paper_state_path, output_dir, "paper_account_state.json")
    if cfg.paper_reset_state or not path.exists():
        return default_state(cfg.paper_initial_cash), path
    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError):
        state = default_state(cfg.paper_initial_cash)
    state.setdefault("cash", float(cfg.paper_initial_cash))
    state.setdefault("positions", {})
    state.setdefault("realized_pnl", 0.0)
    state.setdefault("unrealized_pnl", 0.0)
    state.setdefault("equity", float(state.get("cash", cfg.paper_initial_cash)))
    state.setdefault("last_update", "")
    return state, path


def save_paper_state(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, default=str)


def append_ledger(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    existing = pd.read_csv(path) if path.exists() and path.stat().st_size > 0 else pd.DataFrame()
    combined = pd.concat([existing, frame], ignore_index=True)
    combined.to_csv(path, index=False)
    return combined


def position_snapshot_rows(state: dict[str, Any], timestamp: str) -> list[dict[str, Any]]:
    rows = []
    equity = float(state.get("equity", 0.0) or 0.0)
    for ticker, position in dict(state.get("positions", {})).items():
        market_value = float(position.get("market_value", 0.0) or 0.0)
        rows.append(
            {
                "timestamp": timestamp,
                "ticker": ticker,
                "qty": float(position.get("quantity", 0.0) or 0.0),
                "avg_price": float(position.get("avg_price", 0.0) or 0.0),
                "last_price": float(position.get("last_price", 0.0) or 0.0),
                "market_value": market_value,
                "weight": market_value / equity if equity else 0.0,
            }
        )
    if not rows:
        rows.append(
            {
                "timestamp": timestamp,
                "ticker": "",
                "qty": 0.0,
                "avg_price": 0.0,
                "last_price": 0.0,
                "market_value": 0.0,
                "weight": 0.0,
            }
        )
    return rows
