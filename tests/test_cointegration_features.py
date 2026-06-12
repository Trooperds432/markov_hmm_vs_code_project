from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.feature_blocks.research_public_features import (
    build_research_public_features,
)


def test_cointegration_spread_features_are_finite_on_synthetic_pair():
    rng = np.random.default_rng(44)
    dates = pd.bdate_range("2020-01-01", periods=320)
    peer = np.cumsum(rng.normal(0.0, 0.01, len(dates)))
    asset = 1.2 * peer + rng.normal(0.0, 0.03, len(dates))
    frame = pd.DataFrame(
        {
            "Close": 100 * np.exp(asset),
            "Benchmark_Close": 100 * np.exp(peer),
            "semiconductors_Close": 90 * np.exp(peer),
            "ret_1": pd.Series(asset, index=dates).diff(),
        },
        index=dates,
    )
    cfg = PipelineConfig(
        enable_research_public_features=True,
        research_public_feature_set=("cointegration_spread",),
        cointegration_window=80,
    )

    result = build_research_public_features(frame, cfg)

    assert result.features["sector_spread_zscore"].dropna().notna().any()
    assert result.features["spread_volatility"].dropna().ge(0.0).all()
    assert "cointegration_pvalue_vs_sector" in result.features
