from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBlockResult, feature_metadata


def build_price_core(df: pd.DataFrame) -> FeatureBlockResult:
    out = pd.DataFrame(index=df.index)
    close = df["Close"].astype(float)
    benchmark = df.get("Benchmark_Close", close).astype(float)
    returns = np.log(close / close.shift(1))
    out["fg_ret_1"] = returns
    out["fg_ret_5"] = np.log(close / close.shift(5))
    out["fg_ret_20"] = np.log(close / close.shift(20))
    out["fg_rolling_mean_20"] = returns.rolling(20).mean().shift(1)
    out["fg_realised_vol_20"] = returns.rolling(20).std(ddof=0).shift(1) * np.sqrt(252)
    out["fg_drawdown_63"] = close / close.rolling(63, min_periods=1).max() - 1.0
    out["fg_trend_slope_20"] = close.pct_change(20).shift(1)
    out["fg_gap_1"] = np.log(df["Open"].astype(float) / close.shift(1))
    out["fg_corr_benchmark_63"] = (
        returns.rolling(63).corr(np.log(benchmark / benchmark.shift(1))).shift(1)
    )
    columns = list(out.columns)
    return FeatureBlockResult(
        name="price_core",
        features=out.replace([np.inf, -np.inf], np.nan),
        metadata=feature_metadata("price_core", columns),
        produced_columns=columns,
    )
