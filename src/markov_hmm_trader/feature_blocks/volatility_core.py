from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBlockResult, feature_metadata


def build_volatility_core(df: pd.DataFrame) -> FeatureBlockResult:
    returns = np.log(df["Close"] / df["Close"].shift(1)).replace([np.inf, -np.inf], np.nan)
    downside = returns.where(returns < 0.0, 0.0)
    features = pd.DataFrame(index=df.index)
    features["fg_realised_vol_63"] = returns.rolling(63).std(ddof=0).shift(1)
    features["fg_vol_ratio_20_63"] = returns.rolling(20).std(ddof=0).shift(1) / features[
        "fg_realised_vol_63"
    ].replace(0.0, np.nan)
    features["fg_downside_vol_20"] = downside.rolling(20).std(ddof=0).shift(1)
    columns = list(features.columns)
    return FeatureBlockResult(
        name="volatility_core",
        features=features,
        metadata=feature_metadata("volatility_core", columns),
        produced_columns=columns,
    )
