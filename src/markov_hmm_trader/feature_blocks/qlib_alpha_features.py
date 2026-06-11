from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBlockResult, feature_metadata


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)

    def slope(values: np.ndarray) -> float:
        if np.isnan(values).any():
            return np.nan
        return float(np.polyfit(x, values, 1)[0])

    return series.rolling(window).apply(slope, raw=True)


def build_alpha158_lite(df: pd.DataFrame) -> FeatureBlockResult:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    returns = np.log(close / close.shift(1))
    volume_ma = volume.rolling(20).mean()
    out = pd.DataFrame(
        {
            "alpha_ret_10": np.log(close / close.shift(10)).shift(1),
            "alpha_ret_60": np.log(close / close.shift(60)).shift(1),
            "alpha_slope_20": _rolling_slope(np.log(close), 20).shift(1),
            "alpha_corr_ret_volume_20": returns.rolling(20).corr(volume.pct_change()).shift(1),
            "alpha_volume_wma_ratio_20": (volume / volume_ma.replace(0.0, np.nan)).shift(1),
            "alpha_volatility_20": returns.rolling(20).std(ddof=0).shift(1),
            "alpha_price_position_60": (
                (close - low.rolling(60).min()) / (high.rolling(60).max() - low.rolling(60).min())
            ).shift(1),
        },
        index=df.index,
    )
    columns = list(out.columns)
    return FeatureBlockResult(
        name="alpha158_lite",
        features=out.replace([np.inf, -np.inf], np.nan),
        metadata=feature_metadata("alpha158_lite", columns),
        produced_columns=columns,
    )
