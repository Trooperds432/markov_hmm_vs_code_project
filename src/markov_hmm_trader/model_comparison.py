from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from .backtest import walk_forward_backtest
from .config import PipelineConfig
from .metrics import add_backtest_metric_columns, make_report


def _baseline_strategy(df: pd.DataFrame, name: str, position: pd.Series, cfg: PipelineConfig) -> dict[str, object]:
    out = df.copy()
    out["position"] = position.reindex(out.index).fillna(0.0)
    out["strategy_ret"] = out["position"].shift(1).fillna(0.0) * out["ret_1"].fillna(0.0)
    out["strategy_equity"] = (1.0 + out["strategy_ret"]).cumprod()
    out["buy_hold_equity"] = (1.0 + out["ret_1"].fillna(0.0)).cumprod()
    stats = make_report(add_backtest_metric_columns(out, cfg), cfg)
    row = stats.to_dict()
    row["model"] = name
    return row


def run_model_comparison(df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    rows = []
    curves = pd.DataFrame(index=df.index)
    state_rows = []

    try:
        hmm_result = walk_forward_backtest(df, cfg)
        hmm_backtest = add_backtest_metric_columns(hmm_result.backtest, cfg)
        hmm_stats = make_report(hmm_backtest, cfg).to_dict()
        hmm_stats["model"] = "hmmlearn_gaussian_hmm"
        rows.append(hmm_stats)
        curves["hmmlearn_gaussian_hmm"] = hmm_backtest["strategy_equity"]
        for state, group in hmm_backtest.groupby("hidden_state"):
            state_rows.append(
                {
                    "model": "hmmlearn_gaussian_hmm",
                    "state": state,
                    "occupancy": float(len(group) / max(len(hmm_backtest), 1)),
                    "forward_return": float(group["ret_1"].shift(-1).mean()),
                }
            )
    except Exception as err:
        rows.append({"model": "hmmlearn_gaussian_hmm", "error": type(err).__name__})

    try:
        returns = df["ret_1"].fillna(0.0).to_numpy().reshape(-1, 1)
        gmm = GaussianMixture(n_components=cfg.n_states, covariance_type="full", random_state=7)
        states = gmm.fit_predict(returns)
        state_returns = pd.Series(df["ret_1"].values, index=df.index).groupby(states).mean()
        best_state = int(state_returns.idxmax())
        position = pd.Series((states == best_state).astype(float), index=df.index)
        row = _baseline_strategy(df, "gaussian_mixture", position, cfg)
        row["aic"] = float(gmm.aic(returns))
        row["bic"] = float(gmm.bic(returns))
        rows.append(row)
        curves["gaussian_mixture"] = (1.0 + position.shift(1).fillna(0.0) * df["ret_1"]).cumprod()
    except Exception as err:
        rows.append({"model": "gaussian_mixture", "error": type(err).__name__})

    trend_position = (df["Close"] > df["Close"].rolling(cfg.slow_ema).mean()).astype(float)
    rows.append(_baseline_strategy(df, "trend_following", trend_position, cfg))
    curves["trend_following"] = (
        1.0 + trend_position.shift(1).fillna(0.0) * df["ret_1"].fillna(0.0)
    ).cumprod()

    rows.append(_baseline_strategy(df, "buy_and_hold", pd.Series(1.0, index=df.index), cfg))
    curves["buy_and_hold"] = (1.0 + df["ret_1"].fillna(0.0)).cumprod()

    rows.append(_baseline_strategy(df, "cash", pd.Series(0.0, index=df.index), cfg))
    curves["cash"] = 1.0

    return {
        "model_comparison": pd.DataFrame(rows),
        "model_comparison_equity_curves": curves,
        "model_state_comparison": pd.DataFrame(state_rows),
    }
