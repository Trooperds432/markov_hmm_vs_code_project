from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from .joins import safe_asof_join


def build_valuation_features(
    base: pd.DataFrame, valuation: pd.DataFrame, cfg: PipelineConfig
) -> pd.DataFrame:
    joined = safe_asof_join(
        base,
        valuation,
        cfg.ticker,
        "date",
        cfg.valuation_publication_lag_days,
    )
    features = pd.DataFrame(index=base.index)
    intrinsic_source = joined.get("graham_intrinsic_value", pd.Series(np.nan, index=base.index))
    close_source = joined.get("close", pd.Series(np.nan, index=base.index))
    intrinsic = pd.to_numeric(intrinsic_source, errors="coerce")
    close = pd.to_numeric(close_source, errors="coerce").fillna(base.get("Close"))
    invalid = intrinsic <= 0.0
    intrinsic = intrinsic.mask(invalid)
    features["graham_intrinsic_value"] = intrinsic
    features["price_to_intrinsic"] = close / intrinsic.replace(0.0, np.nan)
    features["intrinsic_to_price"] = intrinsic / close.replace(0.0, np.nan)
    features["margin_of_safety"] = features["intrinsic_to_price"] - 1.0
    for column in ("ttm_eps", "eps_growth_rate", "bond_yield"):
        features[column] = pd.to_numeric(
            joined.get(column, pd.Series(np.nan, index=base.index)), errors="coerce"
        )
    mean = features["margin_of_safety"].rolling(cfg.valuation_z_window, min_periods=5).mean()
    std = features["margin_of_safety"].rolling(cfg.valuation_z_window, min_periods=5).std(ddof=0)
    features["valuation_zscore"] = (features["margin_of_safety"] - mean) / std.replace(0.0, np.nan)
    features["valuation_percentile"] = (
        features["margin_of_safety"].rolling(cfg.valuation_z_window, min_periods=5).rank(pct=True)
    )
    features["deep_value_flag"] = features["margin_of_safety"].gt(
        float(cfg.valuation_deep_value_threshold)
    )
    features["overvalued_flag"] = features["margin_of_safety"].lt(
        float(cfg.valuation_overvalued_threshold)
    )
    features["valuation_data_age_days"] = joined.get(
        "date_age_days", pd.Series(np.nan, index=base.index)
    )
    return features
