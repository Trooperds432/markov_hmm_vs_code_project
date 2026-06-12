from __future__ import annotations

import numpy as np

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.hypothesis_validation import run_hypothesis_validation
from src.markov_hmm_trader.states import STATE_BEAR, STATE_BULL, STATE_SIDE
from tests.test_research_public_features import _research_frame


def test_hypothesis_validation_schema_is_stable():
    frame = _research_frame()
    frame[STATE_BULL] = np.linspace(0.2, 0.8, len(frame))
    frame[STATE_BEAR] = 1.0 - frame[STATE_BULL]
    frame[STATE_SIDE] = 0.1
    frame["hidden_state"] = np.where(frame[STATE_BULL] > 0.5, STATE_BULL, STATE_BEAR)
    frame["position"] = 0.5
    frame["strategy_ret"] = frame["ret_1"] * 0.5
    frame["strategy_equity"] = (1.0 + frame["strategy_ret"]).cumprod()
    frame["buy_hold_equity"] = (1.0 + frame["ret_1"].fillna(0.0)).cumprod()

    report = run_hypothesis_validation(frame, PipelineConfig())

    assert {
        "test_name",
        "hypothesis",
        "metric_name",
        "metric_value",
        "p_value",
        "alpha",
        "passed",
        "sample_size",
        "window_id",
        "notes",
    }.issubset(report.columns)
    assert {"bull_vs_bear_forward_return", "random_timing_baseline"}.issubset(
        set(report["test_name"])
    )
