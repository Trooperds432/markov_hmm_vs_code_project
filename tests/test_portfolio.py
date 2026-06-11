from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.portfolio import allocate_weights


def test_allocate_weights_respects_caps_cash_buffer_and_budget():
    cfg = PipelineConfig(
        portfolio_mode="top_k",
        portfolio_max_positions=3,
        portfolio_max_weight_per_asset=0.25,
        portfolio_cash_buffer=0.10,
        portfolio_weighting_method="score_vol_inverse",
    )
    scores = pd.Series({"A": 1.0, "B": 0.8, "C": 0.7, "D": 0.6})
    volatility = pd.Series({"A": 0.2, "B": 0.1, "C": 0.3, "D": 0.4})

    weights = allocate_weights(scores, volatility, cfg)

    assert len(weights) == 3
    assert weights.sum() <= 0.90
    assert weights.max() <= 0.25


def test_single_asset_portfolio_mode_keeps_top_score_only():
    cfg = PipelineConfig(portfolio_mode="single_asset", portfolio_max_positions=3)
    scores = pd.Series({"A": 0.5, "B": 1.0, "C": 0.1})
    volatility = pd.Series({"A": 0.2, "B": 0.2, "C": 0.2})

    weights = allocate_weights(scores, volatility, cfg)

    assert list(weights.index) == ["B"]


def test_risk_budgeted_weights_respect_cash_cap_and_target_vol():
    cfg = PipelineConfig(
        portfolio_mode="risk_budgeted",
        portfolio_max_positions=3,
        portfolio_max_weight_per_asset=0.6,
        portfolio_cash_buffer=0.10,
        portfolio_target_vol=0.20,
    )
    scores = pd.Series({"A": 1.0, "B": 0.9, "C": 0.8})
    volatility = pd.Series({"A": 0.05, "B": 0.05, "C": 0.05})

    weights = allocate_weights(scores, volatility, cfg)

    assert len(weights) == 3
    assert weights.sum() <= 0.90
    assert weights.max() <= 0.6
    assert weights.sum() < 0.90
