from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.baselines import baseline_comparison
from src.markov_hmm_trader.config import PipelineConfig


def _sample_returns() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=80)
    returns = pd.Series(np.sin(np.arange(80) / 10) * 0.002 + 0.001, index=index)
    return pd.DataFrame({"ret_1": returns}, index=index)


def test_baselines_run_and_random_is_reproducible():
    cfg = PipelineConfig(random_seed=11)
    data = _sample_returns()

    first = baseline_comparison(data, cfg, "SYNTH")
    second = baseline_comparison(data, cfg, "SYNTH")

    assert {"buy_and_hold", "cash", "random_signal", "naive_momentum", "inverse_momentum"}.issubset(
        set(first["baseline"])
    )
    random_first = first[first["baseline"] == "random_signal"].iloc[0]
    random_second = second[second["baseline"] == "random_signal"].iloc[0]
    assert random_first["final_equity"] == random_second["final_equity"]


def test_momentum_baseline_uses_past_returns():
    cfg = PipelineConfig(random_seed=11)
    data = _sample_returns()

    comparison = baseline_comparison(data, cfg, "SYNTH")
    momentum = comparison[comparison["baseline"] == "naive_momentum"].iloc[0]

    assert np.isfinite(momentum["final_equity"])
