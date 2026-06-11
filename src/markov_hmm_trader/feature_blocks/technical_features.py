from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBlockResult, feature_metadata


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(window).mean()
    losses = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gains / losses.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def build_technical_features(df: pd.DataFrame) -> FeatureBlockResult:
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
    returns = close.pct_change()
    middle = close.rolling(20).mean()
    width = close.rolling(20).std(ddof=0) * 4.0 / middle.replace(0.0, np.nan)
    trend_range = (high.rolling(14).max() - low.rolling(14).min()) / close.replace(0.0, np.nan)
    obv = _obv(close, volume)
    out = pd.DataFrame(
        {
            "tech_rsi_14": _rsi(close, 14).shift(1),
            "tech_vwap_distance_20": (close / vwap - 1.0).shift(1),
            "tech_obv_slope_20": obv.pct_change(20).shift(1),
            "tech_trend_strength_14": trend_range.shift(1),
            "tech_bollinger_width_20": width.shift(1),
            "tech_clenow_momentum_90": (
                close.pct_change(90) / returns.rolling(90).std(ddof=0)
            ).shift(1),
        },
        index=df.index,
    )
    columns = list(out.columns)
    return FeatureBlockResult(
        name="technical",
        features=out.replace([np.inf, -np.inf], np.nan),
        metadata=feature_metadata("technical", columns),
        produced_columns=columns,
    )
