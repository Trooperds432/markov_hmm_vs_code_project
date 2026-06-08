from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def cagr(equity: pd.Series, bars_per_year: int) -> float:
    equity = equity.dropna()
    if len(equity) < 2:
        return np.nan

    years = len(equity) / bars_per_year
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def sharpe(returns: pd.Series, bars_per_year: int) -> float:
    returns = returns.dropna()
    std = returns.std(ddof=0)

    if std == 0 or np.isnan(std):
        return np.nan

    return float(returns.mean() / std * np.sqrt(bars_per_year))


def trade_stats(df: pd.DataFrame) -> dict[str, float]:
    pos = df["position"].fillna(0.0)
    in_trade = pos.shift(1).fillna(0.0) != 0.0
    trade_id = (in_trade != in_trade.shift(1).fillna(False)).cumsum()

    trade_rets = []
    for _, group in df[in_trade].groupby(trade_id[in_trade]):
        trade_rets.append((1.0 + group["strategy_ret"]).prod() - 1.0)

    if not trade_rets:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "avg_trade": np.nan,
        }

    tr = np.array(trade_rets)
    gross_profit = tr[tr > 0].sum()
    gross_loss = abs(tr[tr < 0].sum())

    return {
        "trades": int(len(tr)),
        "win_rate": float((tr > 0).mean()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        "avg_trade": float(tr.mean()),
    }


def make_report(df: pd.DataFrame, cfg: PipelineConfig) -> pd.Series:
    strat = df["strategy_equity"].dropna()
    buy_hold = df["buy_hold_equity"].reindex(strat.index).dropna()
    strat = strat.reindex(buy_hold.index)

    stats = {
        "ticker": cfg.ticker,
        "benchmark": cfg.benchmark,
        "strategy_total_return": strat.iloc[-1] - 1.0,
        "buy_hold_total_return": buy_hold.iloc[-1] - 1.0,
        "strategy_cagr": cagr(strat, cfg.bars_per_year),
        "buy_hold_cagr": cagr(buy_hold, cfg.bars_per_year),
        "strategy_max_drawdown": max_drawdown(strat),
        "buy_hold_max_drawdown": max_drawdown(buy_hold),
        "strategy_sharpe": sharpe(df["strategy_ret"].reindex(strat.index), cfg.bars_per_year),
        "avg_abs_position": df["position"].abs().mean(),
    }

    stats.update(trade_stats(df))
    return pd.Series(stats)
