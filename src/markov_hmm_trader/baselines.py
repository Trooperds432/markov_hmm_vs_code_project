from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .metrics import cagr, calmar, max_drawdown, sharpe, sortino, turnover
from .states import STATE_BEAR, STATE_BULL

BASELINE_COLUMNS = [
    "baseline",
    "symbol",
    "total_return",
    "CAGR",
    "sharpe",
    "Sortino",
    "max_drawdown",
    "Calmar",
    "turnover",
    "trade_count",
    "win_rate",
    "exposure",
    "profit_factor",
    "final_equity",
    "reason",
    "created_at_utc",
]


def _profit_factor(returns: pd.Series) -> float:
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else np.nan
    return float(gains / losses)


def _baseline_metrics(
    name: str,
    returns: pd.Series,
    position: pd.Series,
    cfg: PipelineConfig,
    symbol: str,
    created_at_utc: str,
) -> dict[str, object]:
    returns = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    position = position.reindex(returns.index).fillna(0.0)
    equity = (1.0 + returns).cumprod()
    cagr_value = cagr(equity, cfg.bars_per_year)
    max_dd = max_drawdown(equity)
    trades = int((position.diff().abs().fillna(position.abs()) > 0).sum())
    active_returns = returns[position.shift(1).fillna(0.0) != 0.0]
    return {
        "baseline": name,
        "symbol": symbol,
        "total_return": float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan,
        "CAGR": cagr_value,
        "sharpe": sharpe(returns, cfg.bars_per_year),
        "Sortino": sortino(returns, cfg.bars_per_year),
        "max_drawdown": max_dd,
        "Calmar": calmar(cagr_value, max_dd),
        "turnover": turnover(position),
        "trade_count": trades,
        "win_rate": float((active_returns > 0).mean()) if not active_returns.empty else np.nan,
        "exposure": float(position.abs().mean()) if not position.empty else np.nan,
        "profit_factor": _profit_factor(active_returns),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else np.nan,
        "reason": "",
        "created_at_utc": created_at_utc,
    }


def baseline_comparison(
    df: pd.DataFrame, cfg: PipelineConfig, symbol: str | None = None
) -> pd.DataFrame:
    symbol = symbol or cfg.ticker
    created_at_utc = datetime.now(UTC).replace(microsecond=0).isoformat()
    returns = df["ret_1"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows = []

    positions: dict[str, pd.Series] = {
        "buy_and_hold": pd.Series(1.0, index=df.index),
        "cash": pd.Series(0.0, index=df.index),
    }
    rng = np.random.default_rng(int(cfg.random_seed))
    positions["random_signal"] = pd.Series(
        rng.choice([0.0, 1.0], size=len(df), p=[0.5, 0.5]),
        index=df.index,
    )
    momentum = returns.rolling(20, min_periods=2).sum().shift(1).fillna(0.0)
    positions["naive_momentum"] = (momentum > 0).astype(float)
    positions["inverse_momentum"] = (momentum < 0).astype(float)
    if STATE_BULL in df and STATE_BEAR in df:
        positions["observable_markov"] = (df[STATE_BULL] > df[STATE_BEAR]).astype(float)

    for name, position in positions.items():
        strategy_returns = position.shift(1).fillna(0.0) * returns
        rows.append(
            _baseline_metrics(name, strategy_returns, position, cfg, symbol, created_at_utc)
        )
    return pd.DataFrame(rows, columns=BASELINE_COLUMNS)


def write_baseline_comparison(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=BASELINE_COLUMNS).to_csv(path, index=False)
