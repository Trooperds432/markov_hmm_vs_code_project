from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.hmm_model import map_components_to_semantic_states, semantic_probabilities
from src.markov_hmm_trader.states import STATE_BEAR, STATE_BULL, STATE_SIDE


def test_semantic_mapping_supports_five_components():
    model = SimpleNamespace(n_components=5)
    hidden = np.array([0] * 20 + [1] * 20 + [2] * 20 + [3] * 20 + [4] * 20)
    returns = np.r_[
        np.full(20, 0.02),
        np.full(20, 0.01),
        np.full(20, 0.0),
        np.full(20, -0.01),
        np.full(20, -0.02),
    ]
    train = pd.DataFrame(
        {
            "ret_1": returns,
            "alpha_score_long": [5, 4, 2, 1, 0] * 20,
        }
    )

    state_map, diagnostics = map_components_to_semantic_states(
        model,
        train,
        hidden,
        PipelineConfig(n_states=5),
    )

    assert len(diagnostics) == 5
    assert STATE_BULL in state_map.values()
    assert STATE_BEAR in state_map.values()
    assert STATE_SIDE in state_map.values()


def test_semantic_probabilities_aggregates_many_components():
    probs = np.array([[0.10, 0.20, 0.30, 0.15, 0.25]])
    state_map = {
        0: STATE_BULL,
        1: STATE_BULL,
        2: STATE_SIDE,
        3: STATE_BEAR,
        4: STATE_SIDE,
    }

    semantic = semantic_probabilities(probs, state_map)

    assert round(semantic.loc[0, STATE_BULL], 6) == 0.30
    assert semantic.loc[0, STATE_BEAR] == 0.15
    assert semantic.loc[0, STATE_SIDE] == 0.55
