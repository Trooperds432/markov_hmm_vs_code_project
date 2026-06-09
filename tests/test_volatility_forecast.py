from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.volatility_forecast import (
    ewma_volatility_forecast,
    realised_volatility_forecast,
    run_volatility_forecasts,
)


def test_realised_and_ewma_volatility_are_shifted():
    returns = pd.Series([0.0, 0.01, -0.01, 0.02, -0.02])
    cfg = PipelineConfig(vol_forecast_horizon_bars=3, ewma_lambda=0.94)

    realised = realised_volatility_forecast(returns, cfg)
    ewma = ewma_volatility_forecast(returns, cfg)

    assert pd.isna(realised.iloc[2])
    assert ewma.iloc[0] == 0.0
    assert ewma.iloc[-1] > 0.0


def test_volatility_forecast_contract_and_multiplier_bounds():
    dates = pd.bdate_range("2020-01-01", periods=80)
    df = pd.DataFrame(
        {"ret_1": np.random.default_rng(2).normal(0.0, 0.01, len(dates))},
        index=dates,
    )
    cfg = PipelineConfig(vol_model_candidates=("realised_vol", "ewma_vol", "garch", "arch"))

    output = run_volatility_forecasts(df, cfg)

    assert {"date", "vol_model", "forecast_sigma", "vol_risk_multiplier", "error"}.issubset(output.columns)
    assert output["vol_risk_multiplier"].between(0.0, 1.0).all()
    assert set(output["vol_model"]) == {"realised_vol", "ewma_vol", "garch", "arch"}
