from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .metrics import max_drawdown, sharpe


def _path_metrics(
    path_id: int,
    path_returns: pd.Series,
    baseline_type: str,
    bars_per_year: int,
) -> dict[str, float | int | str]:
    equity = (1.0 + path_returns).cumprod()
    return {
        "path_id": path_id,
        "baseline_type": baseline_type,
        "total_return": float(equity.iloc[-1] - 1.0),
        "sharpe": sharpe(path_returns, bars_per_year),
        "max_drawdown": max_drawdown(equity),
        "final_equity": float(equity.iloc[-1]),
    }


def _strategy_position(backtest: pd.DataFrame) -> pd.Series:
    return backtest.get("position", pd.Series(0.0, index=backtest.index)).fillna(0.0).astype(float)


def _random_positions(
    backtest: pd.DataFrame,
    cfg: PipelineConfig,
    rng: np.random.Generator,
    baseline_type: str,
) -> pd.Series:
    strategy_pos = _strategy_position(backtest)
    exposure = float(strategy_pos.abs().mean())
    signs = rng.choice([-1.0, 0.0, 1.0], size=len(strategy_pos), p=[0.25, 0.50, 0.25])
    if cfg.long_only or (cfg.random_baseline_match_long_only and strategy_pos.min() >= 0.0):
        signs = np.abs(signs)
    position = pd.Series(signs * exposure, index=strategy_pos.index)
    if baseline_type in {"matched_exposure", "matched_turnover", "rebalance", "long_only"}:
        sampled_abs = rng.permutation(strategy_pos.abs().to_numpy())
        position = pd.Series(np.sign(signs) * sampled_abs, index=strategy_pos.index)
        if cfg.long_only or baseline_type == "long_only":
            position = position.abs()
    if baseline_type in {"matched_turnover", "rebalance"}:
        rebalance_mask = strategy_pos.diff().abs().fillna(strategy_pos.abs()) > 1e-12
        position = position.where(rebalance_mask).ffill().fillna(0.0)
    return position.clip(lower=0.0) if cfg.long_only else position


def _random_path_frame(
    backtest: pd.DataFrame,
    cfg: PipelineConfig,
    baseline_type: str,
    n_paths: int | None = None,
) -> pd.DataFrame:
    paths = int(n_paths or cfg.random_baseline_paths)
    returns = backtest["ret_1"].fillna(0.0)
    cost_rate = (cfg.commission_bps + cfg.slippage_bps) / 10_000.0
    seed_offset = sum((idx + 1) * ord(char) for idx, char in enumerate(baseline_type))
    rng = np.random.default_rng(int(cfg.random_baseline_seed) + seed_offset)
    rows = []
    strategy_exposure = float(_strategy_position(backtest).abs().mean())
    strategy_turnover = float(_strategy_position(backtest).diff().abs().fillna(0.0).mean())
    for path_id in range(paths):
        position = _random_positions(backtest, cfg, rng, baseline_type)
        turnover = position.diff().abs().fillna(position.abs())
        path_returns = position.shift(1).fillna(0.0) * returns - turnover * cost_rate
        row = _path_metrics(path_id, path_returns, baseline_type, cfg.bars_per_year)
        row["avg_abs_position"] = float(position.abs().mean())
        row["turnover"] = float(turnover.mean())
        row["strategy_avg_abs_position"] = strategy_exposure
        row["strategy_turnover"] = strategy_turnover
        rows.append(row)
    return pd.DataFrame(rows)


def run_random_timing_baseline(
    backtest: pd.DataFrame,
    cfg: PipelineConfig,
    n_paths: int | None = None,
) -> pd.DataFrame:
    return _random_path_frame(backtest, cfg, "simple", n_paths=n_paths)


def run_matched_random_baselines(
    backtest: pd.DataFrame, cfg: PipelineConfig
) -> dict[str, pd.DataFrame]:
    return {
        "random_timing_matched_exposure": _random_path_frame(backtest, cfg, "matched_exposure"),
        "random_timing_matched_turnover": _random_path_frame(backtest, cfg, "matched_turnover"),
        "random_rebalance_baseline": _random_path_frame(backtest, cfg, "rebalance"),
        "random_long_only_baseline": _random_path_frame(backtest, cfg, "long_only"),
        "random_portfolio_baseline": _random_path_frame(backtest, cfg, "portfolio"),
    }


def summarize_random_baseline(
    strategy_stats: pd.Series,
    random_paths: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    hmm_return = float(strategy_stats.get("strategy_total_return", np.nan))
    hmm_sharpe = float(strategy_stats.get("strategy_sharpe", np.nan))
    hmm_dd = float(strategy_stats.get("strategy_max_drawdown", np.nan))
    return pd.DataFrame(
        [
            {
                "hmm_return": hmm_return,
                "random_median_return": float(random_paths["total_return"].median()),
                "random_p05_return": float(random_paths["total_return"].quantile(0.05)),
                "random_p95_return": float(random_paths["total_return"].quantile(0.95)),
                "hmm_sharpe": hmm_sharpe,
                "random_median_sharpe": float(random_paths["sharpe"].median()),
                "hmm_max_drawdown": hmm_dd,
                "random_median_max_drawdown": float(random_paths["max_drawdown"].median()),
                "hmm_return_percentile": float((random_paths["total_return"] <= hmm_return).mean()),
                "hmm_sharpe_percentile": float((random_paths["sharpe"] <= hmm_sharpe).mean()),
                "hmm_drawdown_percentile": float((random_paths["max_drawdown"] <= hmm_dd).mean()),
                "fraction_random_beating_hmm": float(
                    (random_paths["total_return"] > hmm_return).mean()
                ),
                "random_median_turnover": float(
                    random_paths.get("turnover", pd.Series(dtype=float)).median()
                ),
                "random_median_exposure": float(
                    random_paths.get("avg_abs_position", pd.Series(dtype=float)).median()
                ),
            }
        ]
    )
