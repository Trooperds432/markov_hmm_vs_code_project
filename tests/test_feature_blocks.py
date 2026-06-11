from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.data_providers.synthetic_provider import SyntheticProvider
from src.markov_hmm_trader.feature_blocks.breadth_features import build_breadth_features
from src.markov_hmm_trader.feature_blocks.optional_blocks import build_optional_skip_group
from src.markov_hmm_trader.feature_blocks.price_core import build_price_core
from src.markov_hmm_trader.feature_blocks.qlib_alpha_features import build_alpha158_lite
from src.markov_hmm_trader.feature_blocks.runner import build_configured_feature_groups
from src.markov_hmm_trader.feature_blocks.technical_features import build_technical_features
from src.markov_hmm_trader.features import (
    feature_group_metadata,
    feature_group_warnings,
    make_features,
)


def _raw_frame(periods: int = 260) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=periods)
    close = pd.Series(100.0 * np.exp(np.linspace(0.0, 0.25, periods)), index=dates)
    volume = pd.Series(np.linspace(1_000_000, 2_000_000, periods), index=dates)
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": volume,
            "Benchmark_Close": close * 0.95,
        },
        index=dates,
    )


def test_price_technical_and_alpha_blocks_return_metadata():
    raw = _raw_frame()

    price = build_price_core(raw)
    technical = build_technical_features(raw)
    alpha = build_alpha158_lite(raw)

    assert "fg_ret_20" in price.produced_columns
    assert "tech_rsi_14" in technical.produced_columns
    assert "alpha_ret_60" in alpha.produced_columns
    assert set(["feature_group", "feature", "status"]).issubset(price.metadata.columns)


def test_breadth_feature_group_uses_provider_capability():
    cfg = PipelineConfig(data_provider="synthetic", feature_groups_enabled=("breadth",))
    provider = SyntheticProvider(cfg)

    result = build_breadth_features(_raw_frame(), cfg, provider)

    assert "breadth_percent_positive" in result.features.columns
    assert not result.metadata.empty


def test_optional_group_skip_contract_when_provider_lacks_capability():
    provider = SyntheticProvider(PipelineConfig(data_provider="synthetic"))

    result = build_optional_skip_group("options", provider)

    assert result.features.empty
    assert result.warnings.iloc[0]["reason"] == "provider_lacks_capability"
    assert result.skipped[0]["feature_group"] == "options"


def test_configured_feature_groups_include_metadata_and_warning_rows():
    cfg = PipelineConfig(
        data_provider="synthetic",
        feature_groups_enabled=("price_core", "technical", "alpha158_lite", "breadth", "options"),
    )
    provider = SyntheticProvider(cfg)

    result = build_configured_feature_groups(_raw_frame(), cfg, provider)

    assert {"fg_ret_1", "tech_rsi_14", "alpha_ret_10", "breadth_percent_positive"}.issubset(
        result.features.columns
    )
    assert "options" in set(result.warnings["feature_group"])


def test_make_features_preserves_feature_group_metadata_and_warnings():
    cfg = PipelineConfig(
        data_provider="synthetic",
        slow_ema=30,
        feature_groups_enabled=("price_core", "volatility_core", "options"),
    )

    features = make_features(_raw_frame(), cfg)

    metadata = feature_group_metadata(features)
    warnings = feature_group_warnings(features)
    assert not features.empty
    assert "fg_ret_1" in features.columns
    assert "fg_realised_vol_63" in features.columns
    assert "price_core" in set(metadata["feature_group"])
    assert "options" in set(warnings["feature_group"])
