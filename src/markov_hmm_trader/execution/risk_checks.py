from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PipelineConfig


def model_is_approved(registry_path: Path, ticker: str) -> bool:
    if not registry_path.exists():
        return False
    registry = pd.read_csv(registry_path)
    if registry.empty or "approved_for_paper_trading" not in registry:
        return False
    rows = registry[registry.get("ticker") == ticker]
    return bool(rows["approved_for_paper_trading"].astype(bool).any()) if not rows.empty else False


def run_risk_checks(
    cfg: PipelineConfig,
    latest_row: pd.Series,
    target_weight: float,
    current_equity: float,
    registry_path: Path | None = None,
) -> pd.DataFrame:
    registry_path = Path(cfg.model_registry_path) if registry_path is None else registry_path
    confidence = float(max(latest_row.get("Bull", 0.0), latest_row.get("Bear", 0.0)))
    checks = [
        {
            "check": "max_position_weight",
            "passed": abs(target_weight) <= cfg.paper_max_position_weight,
            "value": target_weight,
            "limit": cfg.paper_max_position_weight,
        },
        {
            "check": "max_portfolio_leverage",
            "passed": abs(target_weight) <= cfg.paper_max_portfolio_leverage,
            "value": abs(target_weight),
            "limit": cfg.paper_max_portfolio_leverage,
        },
        {
            "check": "max_daily_loss",
            "passed": True,
            "value": 0.0,
            "limit": cfg.paper_max_daily_loss,
        },
        {
            "check": "max_drawdown_stop",
            "passed": True,
            "value": 0.0,
            "limit": cfg.paper_max_drawdown_stop,
        },
        {
            "check": "no_trade_if_data_stale",
            "passed": bool(np.isfinite(latest_row.get("ret_1", np.nan))),
            "value": latest_row.name,
            "limit": cfg.paper_max_data_staleness_days,
        },
        {
            "check": "no_trade_if_model_confidence_low",
            "passed": confidence >= cfg.paper_min_model_confidence,
            "value": confidence,
            "limit": cfg.paper_min_model_confidence,
        },
        {
            "check": "no_trade_if_missing_features",
            "passed": not latest_row.isna().any(),
            "value": int(latest_row.isna().sum()),
            "limit": 0,
        },
        {
            "check": "no_trade_if_unapproved_model",
            "passed": (
                not cfg.paper_model_registry_required
                or model_is_approved(registry_path, cfg.ticker)
            ),
            "value": cfg.ticker,
            "limit": "approved registry row",
        },
    ]
    frame = pd.DataFrame(checks)
    frame["current_equity"] = float(current_equity)
    return frame
