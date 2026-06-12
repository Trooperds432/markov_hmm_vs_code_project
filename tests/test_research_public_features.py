from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.feature_blocks.research_public_features import (
    build_research_public_features,
)
from src.markov_hmm_trader.features import feature_columns


def _research_frame(periods: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2020-01-01", periods=periods)
    common = rng.normal(0.0004, 0.01, periods)
    asset = common + rng.normal(0.0, 0.004, periods)
    benchmark = common + rng.normal(0.0, 0.003, periods)
    sector = common * 1.1 + rng.normal(0.0, 0.004, periods)
    close = 100 * np.exp(np.cumsum(asset))
    frame = pd.DataFrame(
        {
            "Close": close,
            "Benchmark_Close": 100 * np.exp(np.cumsum(benchmark)),
            "market_Close": 100 * np.exp(np.cumsum(common)),
            "tech_Close": 80 * np.exp(np.cumsum(common * 0.9)),
            "semiconductors_Close": 90 * np.exp(np.cumsum(sector)),
            "ret_1": asset,
            "ret_5": pd.Series(asset, index=dates).rolling(5).sum().values,
            "ret_20": pd.Series(asset, index=dates).rolling(20).sum().values,
        },
        index=dates,
    )
    return frame


def test_research_public_core_feature_families_exist():
    cfg = PipelineConfig(
        enable_research_public_features=True,
        research_public_feature_set=(
            "stationarity",
            "autoregressive",
            "volatility_clustering",
            "market_residual",
            "correlation_regime",
            "regression_diagnostics",
            "factor_ranks",
            "instability",
        ),
    )

    result = build_research_public_features(_research_frame(), cfg)

    expected = {
        "adf_pvalue_60d",
        "acf_1",
        "ar1_beta_60d",
        "ewma_vol_20d",
        "vol_cluster_score",
        "rolling_beta_spy_60d",
        "market_model_r2",
        "corr_spy_60d",
        "durbin_watson_residuals",
        "residual_tail_risk_score",
        "rank_percentile",
        "parameter_instability_z",
    }
    assert expected.issubset(result.features.columns)
    assert result.metadata["feature_group"].eq("research_public").all()


def test_future_rows_do_not_change_past_research_public_features():
    cfg = PipelineConfig(
        enable_research_public_features=True,
        research_public_feature_set=(
            "autoregressive",
            "volatility_clustering",
            "market_residual",
            "correlation_regime",
        ),
    )
    base = _research_frame()
    perturbed = base.copy()
    perturbed.iloc[180:, perturbed.columns.get_loc("Close")] *= 4.0

    original = build_research_public_features(base, cfg).features
    changed = build_research_public_features(perturbed, cfg).features

    pd.testing.assert_frame_equal(
        original.iloc[:170].replace([np.inf, -np.inf], np.nan),
        changed.iloc[:170].replace([np.inf, -np.inf], np.nan),
        check_dtype=False,
    )


def test_validation_only_columns_are_blocked_from_hmm_features():
    cfg = PipelineConfig(
        use_research_public_hmm_features=True,
        research_public_hmm_features=("acf_1", "top_decile_forward_return"),
    )

    with pytest.raises(ValueError, match="Validation-only"):
        feature_columns(cfg)


def test_alt_data_validation_columns_are_blocked_from_hmm_features():
    cfg = PipelineConfig(
        use_alt_data_hmm_features=True,
        alt_data_hmm_features=("glassdoor_stress_z", "actual_outperformance"),
    )

    with pytest.raises(ValueError, match="Validation-only"):
        feature_columns(cfg)
