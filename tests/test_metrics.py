from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.metrics import (
    annualized_volatility,
    beta_alpha,
    calmar,
    exposure_adjusted_cagr,
    information_ratio,
    sortino,
    tail_risk,
    turnover,
)


def test_annualized_volatility_and_sortino():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01])

    assert annualized_volatility(returns, 4) == returns.std(ddof=0) * 2
    assert sortino(returns, 4) > 0


def test_calmar_and_exposure_adjusted_cagr():
    assert calmar(0.2, -0.1) == 2.0
    assert exposure_adjusted_cagr(0.2, 0.5) == 0.4
    assert np.isnan(exposure_adjusted_cagr(0.2, 0.0))


def test_turnover_from_position_changes():
    position = pd.Series([0.0, 0.5, 1.0, 0.25])

    assert turnover(position) == (0.0 + 0.5 + 0.5 + 0.75) / 4


def test_beta_alpha_and_information_ratio():
    asset = pd.Series([0.01, 0.02, -0.01, 0.03])
    strategy = asset * 0.5 + 0.001

    beta, alpha = beta_alpha(strategy, asset, 252)

    assert round(beta, 6) == 0.5
    assert alpha > 0
    assert information_ratio(strategy, asset, 0.5, 252) > 0


def test_tail_risk_var_and_cvar():
    returns = pd.Series([-0.10, -0.05, 0.01, 0.02, 0.03])
    risk = tail_risk(returns)

    assert risk["var_95"] <= -0.05
    assert risk["cvar_95"] <= risk["var_95"]
    assert risk["worst_day"] == -0.10
