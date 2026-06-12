from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.confidence import apply_statistical_confidence_scaling
from src.markov_hmm_trader.config import PipelineConfig


def test_statistical_confidence_multiplier_clips_and_reduces():
    cfg = PipelineConfig(
        use_statistical_confidence_scaling=True,
        min_statistical_confidence_mult=0.25,
        max_statistical_confidence_mult=1.25,
    )
    row = pd.Series(
        {
            "stationarity_risk_score": 2.0,
            "parameter_instability_z": 2.0,
            "market_model_r2": 0.8,
            "pc1_explained_variance": 0.7,
            "systemic_correlation_score": 2.0,
            "residual_tail_risk_score": 2.0,
        }
    )

    mult = apply_statistical_confidence_scaling(row, cfg)

    assert 0.25 <= mult <= 1.25
    assert mult < 1.0


def test_statistical_confidence_disabled_preserves_multiplier():
    row = pd.Series({"stationarity_risk_score": 999.0})

    assert apply_statistical_confidence_scaling(row, PipelineConfig()) == 1.0
