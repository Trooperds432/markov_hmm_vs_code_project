from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.ensemble import add_ensemble_signals, ensemble_signal_for_row
from src.markov_hmm_trader.states import STATE_BEAR, STATE_BULL, STATE_SIDE


def test_hmm_and_nn_agreement_blocks_disagreement():
    cfg = PipelineConfig(ensemble_mode="hmm_and_nn_agree")
    row = pd.Series(
        {
            STATE_BULL: 0.7,
            STATE_BEAR: 0.1,
            "nn_probability_positive": 0.2,
        }
    )

    assert ensemble_signal_for_row(row, cfg) == 0.0


def test_add_ensemble_signals_creates_diagnostics_columns():
    df = pd.DataFrame(
        {
            STATE_BULL: [0.7],
            STATE_BEAR: [0.1],
            STATE_SIDE: [0.2],
            "nn_signal": [1],
            "nn_probability_positive": [0.8],
        }
    )

    result = add_ensemble_signals(df, PipelineConfig(ensemble_mode="weighted_blend"))

    assert "ensemble_signal" in result
    assert bool(result.loc[0, "agreement"])
