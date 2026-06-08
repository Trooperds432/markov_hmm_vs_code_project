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


def annualized_volatility(returns: pd.Series, bars_per_year: int) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    return float(returns.std(ddof=0) * np.sqrt(bars_per_year))


def sortino(returns: pd.Series, bars_per_year: int, target_return: float = 0.0) -> float:
    """Annualized Sortino ratio using per-bar target returns."""
    returns = returns.dropna()
    if returns.empty:
        return np.nan

    excess = returns - target_return
    downside = excess[excess < 0.0]
    downside_deviation = np.sqrt(np.mean(np.square(downside))) if len(downside) else 0.0
    if downside_deviation == 0.0 or np.isnan(downside_deviation):
        return np.nan
    return float(excess.mean() / downside_deviation * np.sqrt(bars_per_year))


def calmar(cagr_value: float, max_drawdown_value: float) -> float:
    if not np.isfinite(cagr_value) or not np.isfinite(max_drawdown_value):
        return np.nan
    denominator = abs(max_drawdown_value)
    if denominator == 0.0:
        return np.nan
    return float(cagr_value / denominator)


def exposure_adjusted_cagr(strategy_cagr: float, avg_abs_position: float) -> float:
    if not np.isfinite(strategy_cagr) or not np.isfinite(avg_abs_position):
        return np.nan
    if avg_abs_position <= 0.0:
        return np.nan
    return float(strategy_cagr / avg_abs_position)


def turnover(position: pd.Series) -> float:
    position = position.fillna(0.0)
    if position.empty:
        return np.nan
    return float(position.diff().abs().fillna(position.abs()).mean())


def beta_alpha(
    strategy_returns: pd.Series,
    reference_returns: pd.Series,
    bars_per_year: int,
) -> tuple[float, float]:
    aligned = pd.concat([strategy_returns, reference_returns], axis=1).dropna()
    if aligned.empty:
        return np.nan, np.nan

    strategy = aligned.iloc[:, 0].astype(float)
    reference = aligned.iloc[:, 1].astype(float)
    variance = float(np.var(reference, ddof=0))
    if variance == 0.0 or np.isnan(variance):
        return np.nan, np.nan

    covariance = float(np.cov(strategy, reference, ddof=0)[0, 1])
    beta = covariance / variance
    alpha_annualized = (strategy.mean() - beta * reference.mean()) * bars_per_year
    return float(beta), float(alpha_annualized)


def information_ratio(
    strategy_returns: pd.Series,
    asset_returns: pd.Series,
    avg_abs_position: float,
    bars_per_year: int,
) -> float:
    """Information ratio against exposure-matched passive asset returns."""
    aligned = pd.concat([strategy_returns, asset_returns], axis=1).dropna()
    if aligned.empty or not np.isfinite(avg_abs_position):
        return np.nan

    active = aligned.iloc[:, 0] - aligned.iloc[:, 1] * avg_abs_position
    std = active.std(ddof=0)
    if std == 0.0 or np.isnan(std):
        return np.nan
    return float(active.mean() / std * np.sqrt(bars_per_year))


def rolling_sharpe(returns: pd.Series, window: int, bars_per_year: int) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std(ddof=0)
    return mean / std.replace(0.0, np.nan) * np.sqrt(bars_per_year)


def rolling_drawdown(equity: pd.Series, window: int) -> pd.Series:
    rolling_peak = equity.rolling(window, min_periods=1).max()
    return equity / rolling_peak.replace(0.0, np.nan) - 1.0


def tail_risk(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {
            "var_95": np.nan,
            "cvar_95": np.nan,
            "worst_day": np.nan,
            "skew": np.nan,
            "kurtosis": np.nan,
        }

    var_95 = float(returns.quantile(0.05))
    cvar_sample = returns[returns <= var_95]
    return {
        "var_95": var_95,
        "cvar_95": float(cvar_sample.mean()) if not cvar_sample.empty else np.nan,
        "worst_day": float(returns.min()),
        "skew": float(returns.skew()),
        "kurtosis": float(returns.kurtosis()),
    }


def add_backtest_metric_columns(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    if "strategy_ret" in out:
        out["rolling_sharpe_63"] = rolling_sharpe(out["strategy_ret"], 63, cfg.bars_per_year)
        out["rolling_sharpe_252"] = rolling_sharpe(out["strategy_ret"], 252, cfg.bars_per_year)
    if "strategy_equity" in out:
        out["rolling_drawdown_252"] = rolling_drawdown(out["strategy_equity"], 252)
    return out


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
    strategy_returns = df["strategy_ret"].reindex(strat.index)
    asset_returns = df["ret_1"].reindex(strat.index)
    avg_abs_position = float(df["position"].abs().mean())
    strategy_cagr = cagr(strat, cfg.bars_per_year)
    strategy_max_dd = max_drawdown(strat)
    strategy_turnover = turnover(df["position"].reindex(strat.index))
    asset_beta, asset_alpha = beta_alpha(strategy_returns, asset_returns, cfg.bars_per_year)

    benchmark_returns = None
    if "benchmark_ret_1" in df:
        benchmark_returns = df["benchmark_ret_1"].reindex(strat.index)
    elif "bench_ret_1" in df:
        benchmark_returns = df["bench_ret_1"].reindex(strat.index)

    stats = {
        "ticker": cfg.ticker,
        "benchmark": cfg.benchmark,
        "strategy_total_return": strat.iloc[-1] - 1.0,
        "buy_hold_total_return": buy_hold.iloc[-1] - 1.0,
        "strategy_cagr": strategy_cagr,
        "buy_hold_cagr": cagr(buy_hold, cfg.bars_per_year),
        "strategy_max_drawdown": strategy_max_dd,
        "buy_hold_max_drawdown": max_drawdown(buy_hold),
        "strategy_sharpe": sharpe(strategy_returns, cfg.bars_per_year),
        "strategy_annualized_volatility": annualized_volatility(
            strategy_returns,
            cfg.bars_per_year,
        ),
        "strategy_sortino": sortino(strategy_returns, cfg.bars_per_year),
        "strategy_calmar": calmar(strategy_cagr, strategy_max_dd),
        "exposure_adjusted_cagr": exposure_adjusted_cagr(strategy_cagr, avg_abs_position),
        "avg_abs_position": avg_abs_position,
        "turnover": strategy_turnover,
        "annualized_turnover": strategy_turnover * cfg.bars_per_year,
        "beta_to_asset": asset_beta,
        "alpha_to_asset_annualized": asset_alpha,
        "information_ratio": information_ratio(
            strategy_returns,
            asset_returns,
            avg_abs_position,
            cfg.bars_per_year,
        ),
    }

    if benchmark_returns is not None:
        bench_beta, bench_alpha = beta_alpha(strategy_returns, benchmark_returns, cfg.bars_per_year)
        stats["beta_to_benchmark"] = bench_beta
        stats["alpha_to_benchmark_annualized"] = bench_alpha

    stats.update(tail_risk(strategy_returns))
    stats.update(trade_stats(df))
    return pd.Series(stats)
