from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data import load_market_data
from .factor_validation import factor_rank_diagnostics, long_short_spread_diagnostics
from .features import make_features
from .metrics import cagr, max_drawdown, sharpe
from .runner import run_single
from .states import STATE_BEAR, STATE_BULL


def allocate_weights(
    scores: pd.Series,
    volatility: pd.Series,
    cfg: PipelineConfig,
) -> pd.Series:
    mode = str(cfg.portfolio_mode).lower()
    max_positions = 1 if mode == "single_asset" else int(cfg.portfolio_max_positions)
    selected = scores.sort_values(ascending=False).head(max_positions)
    if selected.empty:
        return pd.Series(dtype=float)

    method = "risk_budgeted" if mode == "risk_budgeted" else cfg.portfolio_weighting_method
    if method == "equal":
        raw = pd.Series(1.0, index=selected.index)
    elif method in {"signal_weighted", "score"}:
        raw = selected.clip(lower=0.0)
    elif method in {"risk_budgeted", "risk_parity_simple", "vol_inverse", "inverse_vol"}:
        raw = 1.0 / volatility.reindex(selected.index).replace(0.0, np.nan)
    elif method == "score_vol_inverse":
        vol_inverse = 1.0 / volatility.reindex(selected.index).replace(0.0, np.nan)
        raw = selected.clip(lower=0.0) * vol_inverse
    else:
        raw = 1.0 / volatility.reindex(selected.index).replace(0.0, np.nan)

    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if raw.sum() <= 0.0:
        raw = pd.Series(1.0, index=selected.index)

    weights = raw / raw.sum()
    if method == "risk_budgeted":
        annual_vol = volatility.reindex(selected.index).replace([np.inf, -np.inf], np.nan)
        annual_vol = annual_vol.fillna(annual_vol.median()).fillna(0.0) * np.sqrt(cfg.bars_per_year)
        portfolio_vol = float(np.sqrt(np.square(weights * annual_vol).sum()))
        if portfolio_vol > 0.0:
            vol_scale = min(1.0, float(cfg.portfolio_target_vol) / portfolio_vol)
            weights = weights * vol_scale

    weights = weights.clip(upper=float(cfg.portfolio_max_weight_per_asset))
    max_invested = max(0.0, 1.0 - float(cfg.portfolio_cash_buffer))
    if weights.sum() > max_invested:
        weights = weights / weights.sum() * max_invested
    total = weights.sum()
    if total > 1.0:
        weights = weights / total
    return weights


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    rule = "W-FRI" if frequency == "weekly" else "ME"
    return pd.Series(index, index=index).resample(rule).last().dropna().values


def run_portfolio(cfg: PipelineConfig, output_dir) -> dict[str, pd.DataFrame | pd.Series]:
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_backtests = {}
    signal_rows = []

    for symbol in cfg.portfolio_symbols:
        local_cfg = replace(
            cfg,
            ticker=symbol,
            benchmark=cfg.portfolio_benchmark,
            enable_diagnostics=False,
            enable_stress_tests=False,
            enable_experiment_tracking=False,
        )
        try:
            backtest, _ = run_single(local_cfg, output_dir / symbol, make_plots=False)
            asset_backtests[symbol] = backtest
            latest = backtest.iloc[-1]
            signal_rows.append(
                {
                    "symbol": symbol,
                    "bull_probability": latest.get(STATE_BULL, np.nan),
                    "bear_probability": latest.get(STATE_BEAR, np.nan),
                    "trend_score": latest.get("alpha_score_long", np.nan),
                    "relative_strength": latest.get("rel_ret_20", np.nan),
                }
            )
        except Exception as err:
            signal_rows.append({"symbol": symbol, "error": type(err).__name__})

    if not asset_backtests:
        empty = pd.DataFrame()
        return {
            "portfolio_equity_curve": empty,
            "portfolio_weights": empty,
            "portfolio_trades": empty,
            "portfolio_stats": pd.Series(dtype=float),
            "portfolio_asset_signals": pd.DataFrame(signal_rows),
        }

    returns = pd.DataFrame({symbol: bt["ret_1"] for symbol, bt in asset_backtests.items()}).dropna(
        how="all"
    )
    scores = pd.DataFrame(
        {
            symbol: bt[STATE_BULL].fillna(0.0) - bt[STATE_BEAR].fillna(0.0)
            for symbol, bt in asset_backtests.items()
        }
    ).reindex(returns.index)
    volatility = returns.rolling(63).std(ddof=0).fillna(returns.expanding().std(ddof=0))
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)

    for date in _rebalance_dates(returns.index, cfg.portfolio_rebalance_frequency):
        date = pd.Timestamp(date)
        if date not in returns.index:
            continue
        historical_scores = scores.loc[:date].iloc[-1]
        historical_vol = volatility.loc[:date].iloc[-1]
        allocated = allocate_weights(historical_scores, historical_vol, cfg)
        weights.loc[date] = 0.0
        weights.loc[date, allocated.index] = allocated

    weights = weights.ffill().fillna(0.0)
    portfolio_returns = (weights.shift(1).fillna(0.0) * returns.fillna(0.0)).sum(axis=1)
    equity = (1.0 + portfolio_returns).cumprod()
    equity_curve = pd.DataFrame(
        {
            "portfolio_ret": portfolio_returns,
            "portfolio_equity": equity,
            "holdings": (weights > 0.0).sum(axis=1),
            "concentration": (weights.pow(2)).sum(axis=1),
        }
    )
    trades = weights.diff().abs().fillna(weights.abs())
    stats = pd.Series(
        {
            "portfolio_cagr": cagr(equity, cfg.bars_per_year),
            "portfolio_sharpe": sharpe(portfolio_returns, cfg.bars_per_year),
            "portfolio_max_drawdown": max_drawdown(equity),
            "turnover": float(trades.sum(axis=1).mean()),
            "average_number_of_holdings": float(equity_curve["holdings"].mean()),
            "concentration": float(equity_curve["concentration"].mean()),
            "weights_sum_max": float(weights.sum(axis=1).max()),
            "max_weight": float(weights.max(axis=1).max()),
        }
    )

    outputs = {
        "portfolio_equity_curve": equity_curve,
        "portfolio_weights": weights,
        "portfolio_trades": trades,
        "portfolio_stats": stats,
        "portfolio_asset_signals": pd.DataFrame(signal_rows),
    }
    equity_curve.to_csv(output_dir / "portfolio_equity_curve.csv")
    weights.to_csv(output_dir / "portfolio_weights.csv")
    trades.to_csv(output_dir / "portfolio_trades.csv")
    stats.to_frame("value").to_csv(output_dir / "portfolio_stats.csv")
    outputs["portfolio_asset_signals"].to_csv(
        output_dir / "portfolio_asset_signals.csv", index=False
    )
    if cfg.enable_research_public_features:
        rank_source = equity_curve.copy()
        rank_source["rank_percentile"] = scores.rank(axis=1, pct=True).mean(axis=1)
        rank_source["ret_1"] = portfolio_returns
        factor_rank_diagnostics(rank_source, cfg).to_csv(
            output_dir / "factor_rank_diagnostics.csv",
            index=False,
        )
        long_short_spread_diagnostics(rank_source, cfg).to_csv(
            output_dir / "long_short_spread_diagnostics.csv",
            index=False,
        )
    return outputs


def load_portfolio_market_data(cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    data = {}
    for symbol in cfg.portfolio_symbols:
        local_cfg = replace(cfg, ticker=symbol, benchmark=cfg.portfolio_benchmark)
        data[symbol] = make_features(load_market_data(local_cfg), local_cfg)
    return data
