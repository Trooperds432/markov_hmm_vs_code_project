from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.portfolio import _apply_portfolio_constraints


def test_portfolio_constraints_cap_turnover_and_explain_decisions():
    dates = pd.bdate_range("2020-01-01", periods=80)
    returns = pd.DataFrame(
        {
            "A": np.sin(np.arange(80)) / 100.0,
            "B": np.sin(np.arange(80)) / 100.0,
            "C": np.cos(np.arange(80)) / 100.0,
        },
        index=dates,
    )
    cfg = PipelineConfig(
        portfolio_max_turnover_per_rebalance=0.20,
        portfolio_min_rebalance_change=0.03,
        portfolio_use_correlation_penalty=True,
        portfolio_max_pairwise_corr=0.50,
        portfolio_max_weight_per_asset=0.50,
        portfolio_cash_buffer=0.10,
        portfolio_corr_window=60,
    )
    raw = pd.Series({"A": 0.45, "B": 0.45, "C": 0.0})
    previous = pd.Series({"A": 0.0, "B": 0.0, "C": 0.0})
    scores = pd.Series({"A": 0.9, "B": 0.4, "C": 0.0})

    final, decisions, constraints, risk, corr_rows = _apply_portfolio_constraints(
        dates[-1],
        raw,
        previous,
        scores,
        returns,
        cfg,
        {},
    )

    assert final.sum() <= 0.90
    assert final.diff().abs().sum() <= 1.0
    assert risk["turnover"] <= cfg.portfolio_max_turnover_per_rebalance + 1e-9
    assert decisions
    assert corr_rows
    assert any(row["constraint"] == "max_pairwise_corr" for row in constraints)
