from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.data_providers.synthetic_provider import SyntheticProvider
from src.markov_hmm_trader.feature_blocks.alt_bars import build_alt_bar_features
from src.markov_hmm_trader.feature_blocks.breadth_features import build_breadth_features
from src.markov_hmm_trader.feature_blocks.famafrench_features import derive_famafrench_features
from src.markov_hmm_trader.feature_blocks.fundamental_features import derive_fundamental_features
from src.markov_hmm_trader.feature_blocks.macro_features import derive_macro_features
from src.markov_hmm_trader.feature_blocks.news_features import derive_news_features
from src.markov_hmm_trader.feature_blocks.optional_blocks import build_optional_skip_group
from src.markov_hmm_trader.feature_blocks.options_features import derive_options_features
from src.markov_hmm_trader.feature_blocks.price_core import build_price_core
from src.markov_hmm_trader.feature_blocks.qlib_alpha_features import build_alpha158_lite
from src.markov_hmm_trader.feature_blocks.runner import build_configured_feature_groups
from src.markov_hmm_trader.feature_blocks.technical_features import build_technical_features
from src.markov_hmm_trader.features import (
    feature_group_metadata,
    feature_group_warnings,
    feature_manifest,
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


def test_options_feature_derivation_from_mocked_chain():
    raw = _raw_frame()
    chain = pd.DataFrame(
        {
            "date": [raw.index[100]] * 4,
            "option_type": ["call", "put", "call", "put"],
            "strike": [125, 125, 130, 120],
            "implied_volatility": [0.30, 0.34, 0.32, 0.36],
            "open_interest": [100, 150, 80, 120],
            "volume": [10, 20, 5, 15],
            "delta": [0.25, -0.25, 0.4, -0.4],
            "gamma": [0.01, 0.02, 0.01, 0.02],
            "vega": [0.2, 0.3, 0.2, 0.3],
            "underlying_price": [125, 125, 125, 125],
        }
    )

    features = derive_options_features(chain, raw)

    assert "atm_iv_near" in features
    assert features["put_call_oi_ratio"].dropna().iloc[-1] > 0.0


def test_macro_fundamental_news_famafrench_and_alt_blocks_are_timestamp_safe():
    raw = _raw_frame()
    macro = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=90),
            "10y": np.linspace(4.0, 4.5, 90),
            "2y": np.linspace(3.5, 4.0, 90),
            "short_rate": np.linspace(2.0, 2.2, 90),
            "credit_spread": np.linspace(1.0, 1.2, 90),
        }
    )
    fundamentals = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=8, freq="30B"),
            "revenue": np.linspace(100, 140, 8),
            "net_income": np.linspace(10, 18, 8),
            "debt": np.linspace(20, 25, 8),
            "equity": np.linspace(50, 70, 8),
            "free_cash_flow": np.linspace(8, 14, 8),
            "eps_actual": np.linspace(1.0, 1.4, 8),
            "eps_estimate": np.linspace(0.9, 1.3, 8),
        }
    )
    news = pd.DataFrame(
        {
            "published_at": [raw.index[60], raw.index[61]],
            "headline": ["Company beats earnings with strong growth", "Regulatory probe risk"],
        }
    )
    factors = pd.DataFrame(
        {
            "date": raw.index,
            "Mkt-RF": np.random.default_rng(1).normal(0.0, 0.01, len(raw)),
            "SMB": np.random.default_rng(2).normal(0.0, 0.01, len(raw)),
            "HML": np.random.default_rng(3).normal(0.0, 0.01, len(raw)),
            "MOM": np.random.default_rng(4).normal(0.0, 0.01, len(raw)),
        }
    )

    assert "macro_risk_score" in derive_macro_features(macro, raw)
    assert "quality_score_proxy" in derive_fundamental_features(fundamentals, raw)
    assert "headline_sentiment_mean" in derive_news_features(news, raw)
    assert "factor_residual_return" in derive_famafrench_features(factors, raw)
    assert "volume_profile_concentration_20" in build_alt_bar_features(raw).features


def test_feature_manifest_marks_hmm_columns_and_leakage_status():
    cfg = PipelineConfig(data_provider="synthetic", slow_ema=30)
    features = make_features(_raw_frame(), cfg)

    manifest = feature_manifest(features, cfg)

    assert {"feature", "source_block", "used_in_hmm", "leakage_status"}.issubset(manifest.columns)
    assert manifest.loc[manifest["feature"] == "z_ret_1", "used_in_hmm"].iloc[0]
