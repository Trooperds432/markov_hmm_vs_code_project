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


def test_multi_signal_ensemble_combines_optional_inputs_and_vol_gate():
    cfg = PipelineConfig(
        ensemble_mode="multi_signal",
        hmm_weight=0.5,
        nn_weight=0.2,
        observable_markov_weight=0.2,
        forecast_weight=0.1,
        nn_positive_threshold=0.55,
    )
    row = pd.Series(
        {
            STATE_BULL: 0.8,
            STATE_BEAR: 0.2,
            STATE_SIDE: 0.0,
            "nn_adjusted_probability": 0.8,
            "observable_markov_signal": 1.0,
            "forecast_signal": -1.0,
            "vol_risk_multiplier": 0.5,
        }
    )

    signal = ensemble_signal_for_row(row, cfg)

    assert signal == 0.3
