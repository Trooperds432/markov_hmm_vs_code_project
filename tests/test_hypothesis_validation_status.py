from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.hypothesis_validation import run_hypothesis_validation


def test_hypothesis_validation_uses_status_not_automatic_pass():
    frame = pd.DataFrame(
        {
            "ret_1": [0.0, 0.01, -0.01],
            "Bull": [0.8, 0.8, 0.1],
            "Bear": [0.1, 0.1, 0.8],
            "hidden_state": ["Bull", "Bull", "Bear"],
            "position": [0.0, 0.2, 0.0],
            "strategy_ret": [0.0, 0.002, 0.0],
            "strategy_equity": [1.0, 1.002, 1.002],
        },
        index=pd.bdate_range("2020-01-01", periods=3),
    )

    report = run_hypothesis_validation(frame, PipelineConfig())

    assert {"status", "reason", "comparison_value", "effect_size", "min_required_sample"}.issubset(
        report.columns
    )
    assert "insufficient_sample" in set(report["status"])
    assert not report["passed"].all()
