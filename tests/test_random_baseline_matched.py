from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.random_baselines import run_matched_random_baselines


def test_matched_random_baselines_are_deterministic_and_exposure_matched():
    dates = pd.bdate_range("2020-01-01", periods=120)
    backtest = pd.DataFrame(
        {
            "ret_1": np.sin(np.arange(120)) / 1000.0,
            "position": np.where(np.arange(120) % 5 == 0, 0.5, 0.0),
        },
        index=dates,
    )
    cfg = PipelineConfig(random_baseline_seed=11, random_baseline_paths=20, long_only=True)

    first = run_matched_random_baselines(backtest, cfg)
    second = run_matched_random_baselines(backtest, cfg)

    pd.testing.assert_frame_equal(
        first["random_timing_matched_exposure"],
        second["random_timing_matched_exposure"],
    )
    exposure = first["random_timing_matched_exposure"]["avg_abs_position"].median()
    assert abs(exposure - backtest["position"].abs().mean()) < 0.10
    assert (first["random_long_only_baseline"]["avg_abs_position"] >= 0.0).all()
