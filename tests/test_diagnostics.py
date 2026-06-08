from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.markov_hmm_trader.diagnostics import make_state_stability_row
from src.markov_hmm_trader.hmm_model import HMMFitResult
from src.markov_hmm_trader.states import STATE_BEAR, STATE_BULL, STATE_SIDE


def _fit(state_map):
    model = SimpleNamespace(
        transmat_=np.array([[0.8, 0.2], [0.1, 0.9]]),
        means_=np.array([[0.0, 1.0], [1.0, 0.0]]),
    )
    return HMMFitResult(
        model=model,
        scaler=SimpleNamespace(),
        state_map=state_map,
        best_seed=1,
        train_log_likelihood=-10.0,
        seed_diagnostics=[],
        component_forward_returns={0: 0.01, 1: -0.01},
        component_occupancy={0: 0.5, 1: 0.5},
    )


def test_state_map_flip_detection():
    previous = _fit({0: STATE_BULL, 1: STATE_BEAR})
    current = _fit({0: STATE_SIDE, 1: STATE_BULL})

    row = make_state_stability_row(2, previous, current)

    assert row["map_changed"] is True
    assert row["bull_component_changed"] is True
    assert row["bear_component_changed"] is True
    assert row["transition_matrix_distance"] == 0.0
