from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.markov_hmm_trader.backtest import walk_forward_backtest
from src.markov_hmm_trader.config import PipelineConfig, load_config
from src.markov_hmm_trader.diagnostics import expected_dwell_times, summarize_states
from src.markov_hmm_trader.features import make_features
from src.markov_hmm_trader.hmm_model import semantic_probabilities
from src.markov_hmm_trader.metrics import cagr, max_drawdown
from src.markov_hmm_trader.states import STATE_BEAR, STATE_BULL, STATE_SIDE
from src.markov_hmm_trader.stress import replay_positions


def test_load_config_converts_tuple_fields_and_ignores_unknowns(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "ticker": "SPY",
                "random_seeds": [1, 2],
                "stress_cost_multipliers": [1.5, 2.0],
                "stress_execution_lag_bars": [1, 3],
                "unknown": "ignored",
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.ticker == "SPY"
    assert cfg.random_seeds == (1, 2)
    assert cfg.stress_cost_multipliers == (1.5, 2.0)
    assert cfg.stress_execution_lag_bars == (1, 3)


def test_semantic_probabilities_aggregates_components():
    raw = np.array([[0.2, 0.3, 0.5]])
    probs = semantic_probabilities(raw, {0: STATE_BULL, 1: STATE_SIDE, 2: STATE_SIDE})

    assert probs.loc[0, STATE_BULL] == 0.2
    assert probs.loc[0, STATE_BEAR] == 0.0
    assert probs.loc[0, STATE_SIDE] == 0.8


def test_metric_helpers():
    equity = pd.Series([1.0, 1.2, 0.9])

    assert round(max_drawdown(equity), 6) == -0.25
    assert round(cagr(pd.Series([1.0, 1.21]), bars_per_year=2), 6) == 0.21


def test_expected_dwell_times():
    dwell = expected_dwell_times(np.array([[0.5, 0.5], [0.2, 0.8]]))

    assert round(dwell[0], 6) == 2.0
    assert round(dwell[1], 6) == 5.0


def test_stress_replay_execution_lag():
    cfg = PipelineConfig(commission_bps=0.0, slippage_bps=0.0, bars_per_year=252)
    df = pd.DataFrame(
        {
            "ret_1": [0.0, 0.10],
            "position": [1.0, 1.0],
        }
    )

    base = replay_positions(df, cfg)
    lagged = replay_positions(df, cfg, execution_lag=1)

    assert round(base["strategy_equity"].iloc[-1], 6) == 1.1
    assert round(lagged["strategy_equity"].iloc[-1], 6) == 1.0


def test_state_summary_uses_backtest_and_window_diagnostics():
    backtest = pd.DataFrame(
        {
            "hidden_state": [STATE_BULL, STATE_BEAR, STATE_BULL],
            STATE_BULL: [0.7, 0.2, 0.6],
            STATE_BEAR: [0.1, 0.6, 0.2],
            STATE_SIDE: [0.2, 0.2, 0.2],
            "ret_1": [0.01, -0.02, 0.03],
            "strategy_ret": [0.0, -0.01, 0.02],
        }
    )
    diagnostics = pd.DataFrame(
        {
            f"{STATE_BULL}_expected_dwell": [4.0],
            f"{STATE_BULL}_forward_return": [0.01],
            f"{STATE_BULL}_occupancy": [0.4],
            f"{STATE_BEAR}_expected_dwell": [3.0],
            f"{STATE_BEAR}_forward_return": [-0.01],
            f"{STATE_BEAR}_occupancy": [0.3],
            f"{STATE_SIDE}_expected_dwell": [2.0],
            f"{STATE_SIDE}_forward_return": [0.0],
            f"{STATE_SIDE}_occupancy": [0.3],
        }
    )

    summary = summarize_states(backtest, diagnostics).set_index("state")

    assert summary.loc[STATE_BULL, "decoded_rows"] == 2
    assert round(summary.loc[STATE_BULL, "avg_train_expected_dwell"], 6) == 4.0


def test_walk_forward_backtest_smoke_on_synthetic_data():
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=360)
    returns = rng.normal(0.001, 0.015, size=len(dates))
    benchmark_returns = rng.normal(0.0005, 0.010, size=len(dates))
    close = 100.0 * np.exp(np.cumsum(returns))
    benchmark_close = 100.0 * np.exp(np.cumsum(benchmark_returns))

    raw = pd.DataFrame(
        {
            "Open": close * (1.0 + rng.normal(0.0, 0.002, size=len(dates))),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, size=len(dates)),
            "Benchmark_Close": benchmark_close,
        },
        index=dates,
    )
    cfg = PipelineConfig(
        n_states=2,
        train_window=35,
        retrain_frequency=10,
        n_iter=5,
        tolerance=1e-2,
        covariance_type="diag",
        random_seeds=(1,),
        vol_lookback=10,
        fast_ema=5,
        slow_ema=15,
        rs_fast=5,
        rs_slow=10,
        enable_diagnostics=True,
        enable_stress_tests=False,
    )

    features = make_features(raw, cfg)
    result = walk_forward_backtest(features, cfg)

    assert not result.backtest.empty
    assert not result.window_diagnostics.empty
    assert not result.seed_diagnostics.empty
    assert "strategy_equity" in result.backtest.columns
