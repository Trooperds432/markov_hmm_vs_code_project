from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.features import FACTOR_FEATURE_COLUMNS, feature_columns, make_features


def test_feature_columns_adds_factor_features_when_enabled():
    cfg = PipelineConfig(use_extra_factor_features=True)

    columns = feature_columns(cfg)

    assert "z_ret_1" in columns
    assert "cross_asset_risk_score" in columns


def test_make_features_with_synthetic_extra_factor_data():
    dates = pd.bdate_range("2020-01-01", periods=260)
    close = pd.Series(100.0 + np.arange(len(dates)), index=dates)
    raw = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
            "Benchmark_Close": close * 0.9,
            "market_Close": close * 0.8,
            "semiconductors_Close": close * 1.1,
            "volatility_proxy_Close": 20.0 + np.sin(np.arange(len(dates))),
            "rates_proxy_Close": close * 0.7,
            "credit_proxy_Close": close * 0.6,
            "dollar_proxy_Close": close * 0.5,
            "defensive_proxy_Close": close * 0.4,
            "growth_proxy_Close": close * 1.2,
        },
        index=dates,
    )
    cfg = PipelineConfig(use_extra_factor_features=True, slow_ema=30)

    features = make_features(raw, cfg)

    assert not features.empty
    assert set(FACTOR_FEATURE_COLUMNS).issubset(features.columns)
    assert features[FACTOR_FEATURE_COLUMNS].isna().sum().sum() == 0
