from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.observable_markov import (
    label_observable_states,
    n_step_forecast,
    run_observable_markov,
    stationary_distribution,
    transition_matrix,
)
from src.markov_hmm_trader.states import STATE_ORDER


def test_observable_markov_transition_and_stationary_sum():
    dates = pd.bdate_range("2020-01-01", periods=80)
    returns = pd.Series(np.r_[np.full(40, 0.002), np.full(40, -0.002)], index=dates)
    df = pd.DataFrame({"ret_1": returns, "Close": 100 * np.exp(returns.cumsum())}, index=dates)
    cfg = PipelineConfig(observable_label_window=5, observable_bull_threshold=0.005, observable_bear_threshold=-0.005)

    labels = label_observable_states(df, cfg)
    matrix = transition_matrix(labels)
    stationary = stationary_distribution(matrix)
    forecast = n_step_forecast(matrix, labels.iloc[-1], 5)

    assert set(matrix.index) == set(STATE_ORDER)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert round(float(stationary.sum()), 6) == 1.0
    assert round(float(forecast.sum()), 6) == 1.0


def test_observable_markov_runner_outputs_contracts():
    dates = pd.bdate_range("2020-01-01", periods=80)
    returns = pd.Series(np.random.default_rng(1).normal(0.0, 0.01, len(dates)), index=dates)
    df = pd.DataFrame({"ret_1": returns, "Close": 100 * np.exp(returns.cumsum())}, index=dates)

    outputs = run_observable_markov(df, PipelineConfig(observable_label_window=5))

    assert "observable_markov_transition_matrix" in outputs
    assert "observable_markov_forecasts" in outputs
    assert not outputs["observable_markov_equity_curve"].empty
