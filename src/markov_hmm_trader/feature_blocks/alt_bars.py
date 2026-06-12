from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBlockResult, feature_metadata


def volume_bar_aggregate(df: pd.DataFrame, volume_target: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    buckets = (df["Volume"].cumsum() / max(float(volume_target), 1.0)).astype(int)
    grouped = df.groupby(buckets)
    out = pd.DataFrame(
        {
            "Open": grouped["Open"].first(),
            "High": grouped["High"].max(),
            "Low": grouped["Low"].min(),
            "Close": grouped["Close"].last(),
            "Volume": grouped["Volume"].sum(),
        }
    )
    out.index = grouped.apply(lambda frame: frame.index[-1])
    return out


def range_bar_aggregate(df: pd.DataFrame, range_pct: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    threshold = max(float(range_pct), 1e-6)
    groups = []
    current = 0
    anchor = float(df["Close"].iloc[0])
    for close in df["Close"].astype(float):
        if abs(close / anchor - 1.0) >= threshold:
            current += 1
            anchor = close
        groups.append(current)
    grouped = df.groupby(groups)
    out = pd.DataFrame(
        {
            "Open": grouped["Open"].first(),
            "High": grouped["High"].max(),
            "Low": grouped["Low"].min(),
            "Close": grouped["Close"].last(),
            "Volume": grouped["Volume"].sum(),
        }
    )
    out.index = grouped.apply(lambda frame: frame.index[-1])
    return out


def build_alt_bar_features(df: pd.DataFrame) -> FeatureBlockResult:
    close = df["Close"].astype(float)
    volume = df["Volume"].astype(float)
    rolling_volume = volume.rolling(20).sum()
    recent_range = close.rolling(20).max() - close.rolling(20).min()
    out = pd.DataFrame(index=df.index)
    out["volume_profile_concentration_20"] = (
        volume.rolling(5).sum() / rolling_volume.replace(0.0, np.nan)
    ).shift(1)
    out["close_position_recent_volume_range"] = (
        (close - close.rolling(20).min()) / recent_range.replace(0.0, np.nan)
    ).shift(1)
    out["volume_bar_pressure_proxy"] = (close.pct_change() * volume).rolling(20).sum().shift(1)
    out["range_bar_pressure_proxy"] = (
        (((df["High"] - df["Low"]) / close.replace(0.0, np.nan)) * np.sign(close.diff()))
        .rolling(20)
        .sum()
        .shift(1)
    )
    columns = list(out.columns)
    return FeatureBlockResult(
        name="alt_bars",
        features=out.replace([np.inf, -np.inf], np.nan),
        metadata=feature_metadata("alt_bars", columns),
        produced_columns=columns,
    )
