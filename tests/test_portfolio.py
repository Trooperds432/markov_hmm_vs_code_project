from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.portfolio import allocate_weights


def test_allocate_weights_respects_caps_and_budget():
    cfg = PipelineConfig(portfolio_max_positions=3, portfolio_max_weight_per_asset=0.25)
    scores = pd.Series({"A": 1.0, "B": 0.8, "C": 0.7, "D": 0.6})
    volatility = pd.Series({"A": 0.2, "B": 0.1, "C": 0.3, "D": 0.4})

    weights = allocate_weights(scores, volatility, cfg)

    assert len(weights) == 3
    assert weights.sum() <= 1.0
    assert weights.max() <= 0.25
