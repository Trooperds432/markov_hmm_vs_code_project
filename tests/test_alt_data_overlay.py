from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.alt_data.overlays import (
    adjusted_portfolio_score,
    compute_alt_data_overlay,
)
from src.markov_hmm_trader.backtest import compute_position_for_row
from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.states import STATE_BEAR, STATE_BULL, STATE_SIDE


def _row(**extra) -> pd.Series:
    base = {
        STATE_BULL: 0.7,
        STATE_BEAR: 0.1,
        STATE_SIDE: 0.2,
        "ensemble_signal": 0.6,
        "rv_realised": 0.2,
        "trend_long": True,
        "rs_long": True,
        "trend_short": False,
        "rs_short": False,
        "alpha_score_long": 5,
        "alpha_score_short": 0,
    }
    base.update(extra)
    return pd.Series(base)


def test_alt_data_overlay_scales_but_does_not_create_exposure_from_zero():
    cfg = PipelineConfig(use_alt_data_position_overlay=True)
    overlay = compute_alt_data_overlay(
        _row(glassdoor_stress_z=3.0, wsb_mentions_z_20d=6.0, margin_of_safety=0.5), cfg
    )

    assert overlay["alt_data_position_multiplier"] < 1.0

    no_signal = _row(**{STATE_BULL: 0.1, STATE_BEAR: 0.8, "ensemble_signal": -0.7})
    assert compute_position_for_row(no_signal, 0.0, cfg) == 0.0


def test_valuation_boosts_confirmed_bull_and_portfolio_score_respects_bear():
    cfg = PipelineConfig()
    bull_score = adjusted_portfolio_score(_row(margin_of_safety=0.5), cfg)
    bear_score = adjusted_portfolio_score(
        _row(**{STATE_BULL: 0.1, STATE_BEAR: 0.8, "margin_of_safety": 0.5}), cfg
    )

    assert bull_score > 0
    assert bear_score < 0
