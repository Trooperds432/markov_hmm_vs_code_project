from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.nested_optimiser import generate_nested_windows, score_stats


def test_generate_nested_windows_uses_train_validation_test_order():
    index = pd.bdate_range("2015-01-01", "2022-12-31")

    windows = generate_nested_windows(index, 3, 1, 1, 1)

    assert not windows.empty
    first = windows.iloc[0]
    assert first["train_start"] < first["train_end"]
    assert first["train_end"] == first["validation_start"]
    assert first["validation_end"] == first["test_start"]


def test_score_stats_selects_positive_quality():
    stats = pd.Series(
        {
            "strategy_sharpe": 1.0,
            "strategy_cagr": 0.2,
            "strategy_max_drawdown": -0.1,
        }
    )

    assert score_stats(stats, "balanced") > 0
    assert score_stats(stats, "calmar_first") == 2.0
