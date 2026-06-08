from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def run_markov_regression_benchmark(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    try:
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    except ImportError:
        return pd.DataFrame(
            [
                {
                    "metric": "status",
                    "value": "statsmodels_not_installed",
                }
            ]
        )

    returns = df["ret_1"].replace([np.inf, -np.inf], np.nan).dropna() * 100.0
    if len(returns) <= cfg.n_states * 10:
        return pd.DataFrame([{"metric": "status", "value": "not_enough_rows"}])

    try:
        model = MarkovRegression(
            returns,
            k_regimes=cfg.n_states,
            trend="c",
            switching_variance=True,
        )
        result = model.fit(maxiter=cfg.benchmark_maxiter, disp=False)
    except Exception as err:
        return pd.DataFrame([{"metric": "status", "value": f"fit_failed:{type(err).__name__}"}])

    mle_retvals = getattr(result, "mle_retvals", {}) or {}
    rows = [
        {"metric": "status", "value": "ok"},
        {"metric": "nobs", "value": float(result.nobs)},
        {"metric": "k_regimes", "value": float(cfg.n_states)},
        {"metric": "llf", "value": float(result.llf)},
        {"metric": "aic", "value": float(result.aic)},
        {"metric": "bic", "value": float(result.bic)},
        {"metric": "converged", "value": str(bool(mle_retvals.get("converged", False)))},
    ]
    return pd.DataFrame(rows)
