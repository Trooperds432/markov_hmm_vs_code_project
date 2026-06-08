from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .metrics import cagr, max_drawdown, sharpe

CRISIS_WINDOWS = {
    "dotcom_2000": ("2000-03-01", "2002-10-31"),
    "global_financial_crisis_2008": ("2007-10-01", "2009-03-31"),
    "covid_crash": ("2020-02-15", "2020-04-30"),
    "rate_hike_2022": ("2022-01-01", "2022-12-31"),
    "rate_shock_2023": ("2023-08-01", "2023-10-31"),
    "ai_momentum_drawdown_2023_2024": ("2023-07-01", "2024-04-30"),
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

    for asset_ret, target_pos in zip(out["ret_1"].fillna(0.0), target, strict=False):
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


def _scenario_summary(
    scenario: str, replay: pd.DataFrame, cfg: PipelineConfig
) -> dict[str, float | int | str]:
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
            rows.append(
                _summary_from_returns(
                    f"crisis_{name}", subset["strategy_ret"], cfg, rows=len(subset)
                )
            )

    return pd.DataFrame(rows)


def block_bootstrap_returns(
    returns: pd.Series,
    paths: int,
    block_size: int,
    random_seed: int,
) -> list[pd.Series]:
    values = returns.fillna(0.0).to_numpy(dtype=float)
    n = len(values)
    if n == 0 or paths <= 0:
        return []

    rng = np.random.default_rng(random_seed)
    max_start = max(n - block_size, 0)
    output = []
    for _ in range(paths):
        samples = []
        while len(samples) < n:
            start = int(rng.integers(0, max_start + 1))
            samples.extend(values[start : start + block_size])
        output.append(pd.Series(samples[:n], index=returns.index))
    return output


def _synthetic_ohlcv_from_returns(df: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    close = float(df["Close"].dropna().iloc[0]) * np.exp(returns.fillna(0.0).cumsum())
    synthetic = df[["Open", "High", "Low", "Close", "Volume", "Benchmark_Close"]].copy()
    synthetic["Close"] = close
    synthetic["Open"] = close * 0.999
    synthetic["High"] = close * 1.01
    synthetic["Low"] = close * 0.99
    return synthetic


def make_asset_bootstrap_stress(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    paths = int(max(cfg.stress_asset_bootstrap_paths, 0))
    if paths == 0:
        return pd.DataFrame()

    rows = []
    bootstrapped = block_bootstrap_returns(
        df["ret_1"],
        paths=paths,
        block_size=max(int(cfg.stress_asset_bootstrap_block_size), 1),
        random_seed=int(cfg.stress_random_seed),
    )
    for path_id, path_returns in enumerate(bootstrapped, start=1):
        try:
            if cfg.stress_rerun_hmm_on_bootstrap and {"Open", "High", "Low", "Close"}.issubset(
                df.columns
            ):
                from .backtest import walk_forward_backtest
                from .features import make_features

                synthetic_raw = _synthetic_ohlcv_from_returns(df, path_returns)
                synthetic_features = make_features(synthetic_raw, cfg)
                result = walk_forward_backtest(synthetic_features, cfg)
                summary = _summary_from_returns(
                    f"asset_bootstrap_{path_id}",
                    result.backtest["strategy_ret"],
                    cfg,
                    rows=len(result.backtest),
                )
            else:
                replay_input = df.copy()
                replay_input["ret_1"] = path_returns
                summary = _scenario_summary(
                    f"asset_bootstrap_{path_id}",
                    replay_positions(replay_input, cfg),
                    cfg,
                )
            summary["path_id"] = path_id
            rows.append(summary)
        except Exception as err:
            rows.append(
                {
                    "scenario": f"asset_bootstrap_{path_id}",
                    "path_id": path_id,
                    "error": type(err).__name__,
                }
            )
    return pd.DataFrame(rows)


def make_cost_slippage_stress(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    for commission in cfg.stress_commission_bps_values:
        for slippage in cfg.stress_slippage_bps_values:
            local_cfg = replace(
                cfg,
                commission_bps=float(commission),
                slippage_bps=float(slippage),
            )
            replay = replay_positions(df, local_cfg)
            summary = _scenario_summary("cost_slippage_grid", replay, local_cfg)
            summary["commission_bps"] = float(commission)
            summary["slippage_bps"] = float(slippage)
            rows.append(summary)
    return pd.DataFrame(rows)


def make_missing_data_stress(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.stress_random_seed)
    rows = []
    for probability in cfg.stress_missing_data_probabilities:
        stressed = df.copy()
        mask = rng.random(len(stressed)) < float(probability)
        stressed.loc[mask, "position"] = np.nan
        stressed["position"] = stressed["position"].ffill().fillna(0.0)
        replay = replay_positions(stressed, cfg)
        summary = _scenario_summary("missing_data", replay, cfg)
        summary["missing_probability"] = float(probability)
        summary["missing_rows"] = int(mask.sum())
        rows.append(summary)
    return pd.DataFrame(rows)


def make_crisis_window_stress(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    windows = dict(CRISIS_WINDOWS)
    for name, values in cfg.custom_crisis_windows.items():
        windows[name] = (values.get("start", ""), values.get("end", ""))

    min_rows = max(20, int(cfg.stress_bootstrap_block_size))
    for name, (start, end) in windows.items():
        if not start or not end:
            continue
        subset = df.loc[start:end]
        if len(subset) >= min_rows:
            row = _summary_from_returns(name, subset["strategy_ret"], cfg, rows=len(subset))
            row["start"] = start
            row["end"] = end
            rows.append(row)
    return pd.DataFrame(rows)
