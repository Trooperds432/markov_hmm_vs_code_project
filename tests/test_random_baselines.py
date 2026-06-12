from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.random_baselines import (
    run_random_timing_baseline,
    summarize_random_baseline,
)


def test_random_baseline_is_deterministic_and_percentiles_are_bounded():
    dates = pd.bdate_range("2020-01-01", periods=80)
    backtest = pd.DataFrame(
        {"ret_1": np.full(len(dates), 0.001), "position": 0.5},
        index=dates,
    )
    cfg = PipelineConfig(random_baseline_seed=123, random_baseline_paths=20)

    first = run_random_timing_baseline(backtest, cfg)
    second = run_random_timing_baseline(backtest, cfg)
    summary = summarize_random_baseline(
        pd.Series(
            {
                "strategy_total_return": 0.01,
                "strategy_sharpe": 0.5,
                "strategy_max_drawdown": -0.02,
            }
        ),
        first,
        cfg,
    )

    pd.testing.assert_frame_equal(first, second)
    assert {"hmm_return_percentile", "fraction_random_beating_hmm"}.issubset(summary.columns)
    assert summary["hmm_return_percentile"].between(0.0, 1.0).all()
