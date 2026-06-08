from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .metrics import cagr, max_drawdown, sharpe


CRISIS_WINDOWS = {
    "covid_crash": ("2020-02-15", "2020-04-30"),
    "rate_hike_2022": ("2022-01-01", "2022-12-31"),
    "rate_shock_2023": ("2023-08-01", "2023-10-31"),
}


def replay_positions(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    cost_multiplier: float = 1.0,
    execution_lag: int = 0,
) -> pd.DataFrame:
    out = df.copy()
    target = out["position"].fillna(0.0)
    if execution_lag > 0:
        target = target.shift(execution_lag).fillna(0.0)

    cost_rate = (cfg.commission_bps + cfg.slippage_bps) / 10_000.0 * cost_multiplier
    prev_pos = 0.0
    equity = 1.0
    equity_curve = []
    strategy_returns = []

    for asset_ret, target_pos in zip(out["ret_1"].fillna(0.0), target):
        turnover = abs(float(target_pos) - prev_pos)
        strategy_ret = prev_pos * float(asset_ret) - turnover * cost_rate
        equity *= 1.0 + strategy_ret
        strategy_returns.append(strategy_ret)
        equity_curve.append(equity)
        prev_pos = float(target_pos)

    out["position"] = target
    out["strategy_ret"] = strategy_returns
    out["strategy_equity"] = equity_curve
    out["buy_hold_equity"] = (1.0 + out["ret_1"].fillna(0.0)).cumprod()
    return out


def _summary_from_returns(
    scenario: str,
    returns: pd.Series,
    cfg: PipelineConfig,
    rows: int | None = None,
) -> dict[str, float | int | str]:
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    if equity.empty:
        return {
            "scenario": scenario,
            "rows": 0 if rows is None else rows,
            "total_return": np.nan,
            "cagr": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
        }

    return {
        "scenario": scenario,
        "rows": int(len(returns) if rows is None else rows),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr(equity, cfg.bars_per_year),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe(returns, cfg.bars_per_year),
    }


def _scenario_summary(scenario: str, replay: pd.DataFrame, cfg: PipelineConfig) -> dict[str, float | int | str]:
    return _summary_from_returns(scenario, replay["strategy_ret"], cfg, rows=len(replay))


def _block_bootstrap_summary(df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, float | int | str]:
    returns = df["strategy_ret"].fillna(0.0).to_numpy(dtype=float)
    n = len(returns)
    block_size = max(int(cfg.stress_bootstrap_block_size), 1)
    n_paths = max(int(cfg.stress_bootstrap_paths), 0)

    row: dict[str, float | int | str] = {
        "scenario": "bootstrap_blocks",
        "rows": n,
        "paths": n_paths,
        "block_size": block_size,
    }

    if n == 0 or n_paths == 0:
        return row

    rng = np.random.default_rng(cfg.stress_random_seed)
    max_start = max(n - block_size, 0)
    totals = []
    cagrs = []
    drawdowns = []
    sharpes = []

    for _ in range(n_paths):
        samples = []
        while len(samples) < n:
            start = int(rng.integers(0, max_start + 1))
            samples.extend(returns[start : start + block_size])
        path_returns = pd.Series(samples[:n])
        summary = _summary_from_returns("bootstrap_path", path_returns, cfg)
        totals.append(summary["total_return"])
        cagrs.append(summary["cagr"])
        drawdowns.append(summary["max_drawdown"])
        sharpes.append(summary["sharpe"])

    for name, values in {
        "total_return": totals,
        "cagr": cagrs,
        "max_drawdown": drawdowns,
        "sharpe": sharpes,
    }.items():
        clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if clean.size:
            row[f"{name}_p05"] = float(np.quantile(clean, 0.05))
            row[f"{name}_median"] = float(np.quantile(clean, 0.50))
            row[f"{name}_p95"] = float(np.quantile(clean, 0.95))

    return row


def make_stress_tests(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    rows.append(_scenario_summary("base_replay", replay_positions(df, cfg), cfg))

    for multiplier in cfg.stress_cost_multipliers:
        replay = replay_positions(df, cfg, cost_multiplier=float(multiplier))
        rows.append(_scenario_summary(f"cost_x{float(multiplier):g}", replay, cfg))

    for lag in cfg.stress_execution_lag_bars:
        replay = replay_positions(df, cfg, execution_lag=int(lag))
        rows.append(_scenario_summary(f"execution_lag_{int(lag)}", replay, cfg))

    rows.append(_block_bootstrap_summary(df, cfg))

    min_crisis_rows = max(20, int(cfg.stress_bootstrap_block_size))
    for name, (start, end) in CRISIS_WINDOWS.items():
        subset = df.loc[start:end]
        if len(subset) >= min_crisis_rows:
            rows.append(_summary_from_returns(f"crisis_{name}", subset["strategy_ret"], cfg, rows=len(subset)))

    return pd.DataFrame(rows)
