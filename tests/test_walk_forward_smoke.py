from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.backtest import walk_forward_backtest
from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.features import make_features
from src.markov_hmm_trader.states import STATE_BEAR, STATE_BULL, STATE_SIDE


def test_walk_forward_smoke_on_synthetic_data():
    dates = pd.bdate_range("2020-01-01", periods=420)
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)
    close = 100.0 * np.exp(returns.cumsum())
    benchmark = 100.0 * np.exp((returns * 0.8 + 0.0001).cumsum())
    raw = pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000 + rng.integers(0, 1000, len(dates)),
            "Benchmark_Close": benchmark,
        },
        index=dates,
    )
    cfg = PipelineConfig(
        train_window=60,
        retrain_frequency=40,
        n_iter=5,
        covariance_type="diag",
        random_seeds=(7,),
        enable_hmm_model_selection=False,
        enable_nn_confirmation=False,
        enable_diagnostics=False,
        enable_stress_tests=False,
        fast_ema=5,
        slow_ema=15,
    )

    features = make_features(raw, cfg)
    result = walk_forward_backtest(features, cfg)

    assert not result.backtest.empty
    assert {STATE_BULL, STATE_BEAR, STATE_SIDE}.issubset(result.backtest.columns)
    assert "position" in result.backtest.columns
    assert {
        "decoded_state_viterbi",
        "decoded_state_map",
        "state_decode_disagreement",
    }.issubset(result.backtest.columns)
