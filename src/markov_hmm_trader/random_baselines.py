from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .metrics import max_drawdown, sharpe


def run_random_timing_baseline(
    backtest: pd.DataFrame,
    cfg: PipelineConfig,
    n_paths: int | None = None,
) -> pd.DataFrame:
    paths = int(n_paths or cfg.random_baseline_paths)
    returns = backtest["ret_1"].fillna(0.0)
    exposure = float(backtest.get("position", pd.Series(0.0, index=backtest.index)).abs().mean())
    rng = np.random.default_rng(int(cfg.random_baseline_seed))
    rows = []
    for path_id in range(paths):
        signs = rng.choice([-1.0, 0.0, 1.0], size=len(returns), p=[0.25, 0.50, 0.25])
        position = pd.Series(signs * exposure, index=returns.index)
        path_returns = position.shift(1).fillna(0.0) * returns
        equity = (1.0 + path_returns).cumprod()
        rows.append(
            {
                "path_id": path_id,
                "total_return": float(equity.iloc[-1] - 1.0),
                "sharpe": sharpe(path_returns, cfg.bars_per_year),
                "max_drawdown": max_drawdown(equity),
                "final_equity": float(equity.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


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
            }
        ]
    )
