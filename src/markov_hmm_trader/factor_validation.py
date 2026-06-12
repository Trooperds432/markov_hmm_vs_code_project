from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def _forward_return(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
    if "Close" in df:
        return np.log(df["Close"].shift(-horizon) / df["Close"])
    return (
        df.get("ret_1", pd.Series(index=df.index, dtype=float))
        .shift(-horizon)
        .rolling(
            horizon,
            min_periods=1,
        )
        .sum()
    )


def factor_rank_diagnostics(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rank = df.get("rank_percentile", pd.Series(0.5, index=df.index, dtype=float))
    fwd_1m = _forward_return(df, 20)
    fwd_3m = _forward_return(df, 60)
    valid = pd.DataFrame({"rank": rank, "fwd_1m": fwd_1m, "fwd_3m": fwd_3m}).dropna()
    if valid.empty:
        return pd.DataFrame(
            [
                {
                    "factor": "momentum_rank",
                    "top_decile_forward_return": np.nan,
                    "bottom_decile_forward_return": np.nan,
                    "long_short_spread_return": np.nan,
                    "factor_spread_zscore": np.nan,
                    "factor_hit_rate": np.nan,
                    "factor_decay_1m": np.nan,
                    "factor_decay_3m": np.nan,
                    "sample_size": 0,
                    "notes": "insufficient_rank_data",
                }
            ]
        )
    top = valid[valid["rank"] >= valid["rank"].quantile(0.9)]
    bottom = valid[valid["rank"] <= valid["rank"].quantile(0.1)]
    spread_series = top["fwd_1m"].reset_index(drop=True) - bottom["fwd_1m"].reset_index(drop=True)
    spread = float(top["fwd_1m"].mean() - bottom["fwd_1m"].mean())
    spread_std = float(spread_series.std(ddof=0)) if len(spread_series) else np.nan
    return pd.DataFrame(
        [
            {
                "factor": "momentum_rank",
                "top_decile_forward_return": float(top["fwd_1m"].mean()),
                "bottom_decile_forward_return": float(bottom["fwd_1m"].mean()),
                "long_short_spread_return": spread,
                "factor_spread_zscore": (
                    spread / spread_std if spread_std and spread_std > 0 else np.nan
                ),
                "factor_hit_rate": float((valid["rank"].gt(0.5) == valid["fwd_1m"].gt(0)).mean()),
                "factor_decay_1m": float(valid["rank"].corr(valid["fwd_1m"])),
                "factor_decay_3m": float(valid["rank"].corr(valid["fwd_3m"])),
                "sample_size": int(len(valid)),
                "notes": "ex_post_forward_return_validation_not_hmm_input",
            }
        ]
    )


def long_short_spread_diagnostics(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    diagnostics = factor_rank_diagnostics(df, cfg).copy()
    diagnostics.insert(0, "spread_name", "rank_top_minus_bottom")
    return diagnostics
