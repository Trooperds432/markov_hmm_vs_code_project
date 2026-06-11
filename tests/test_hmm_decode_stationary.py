from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.markov_hmm_trader.diagnostics import (
    make_decode_comparison_rows,
    make_stationary_distribution_rows,
)
from src.markov_hmm_trader.hmm_model import HMMFitResult
from src.markov_hmm_trader.states import STATE_BEAR, STATE_BULL, STATE_ORDER


def _dummy_fit() -> HMMFitResult:
    model = SimpleNamespace(
        transmat_=np.array([[0.8, 0.2], [0.1, 0.9]]),
        get_stationary_distribution=lambda: np.array([1 / 3, 2 / 3]),
    )
    return HMMFitResult(
        model=model,
        scaler=SimpleNamespace(),
        state_map={0: STATE_BULL, 1: STATE_BEAR},
        best_seed=1,
        train_log_likelihood=-1.0,
        seed_diagnostics=[],
        component_forward_returns={0: 0.01, 1: -0.01},
        component_occupancy={0: 0.4, 1: 0.6},
    )


def test_stationary_distribution_rows_sum_to_one():
    rows = make_stationary_distribution_rows(1, _dummy_fit())

    assert round(sum(row["stationary_probability"] for row in rows), 6) == 1.0
    assert round(sum(row["probability"] for row in rows), 6) == 1.0
    assert {"state", "model_type", "n_states", "covariance_type", "seed", "n_mix"}.issubset(rows[0])


def test_decode_comparison_rows_align_and_flag_bool():
    probs = pd.DataFrame([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]], columns=STATE_ORDER)
    rows = make_decode_comparison_rows(
        1,
        pd.bdate_range("2020-01-01", periods=2),
        np.array([0, 1]),
        np.array([0, 0]),
        probs,
        _dummy_fit(),
    )

    assert len(rows) == 2
    assert isinstance(rows[1]["state_decode_disagreement"], bool)
    assert rows[1]["state_decode_disagreement"] is True
