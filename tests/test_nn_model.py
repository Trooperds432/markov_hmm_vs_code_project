from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.nn_model import add_nn_confirmation, make_forward_labels


def test_forward_labels_use_future_return():
    df = pd.DataFrame({"Close": [100.0, 110.0, 121.0]})

    labeled = make_forward_labels(df, horizon_bars=1)

    assert round(labeled.loc[0, "target_forward_return"], 6) == round(0.0953101798, 6)
    assert labeled.loc[0, "target_positive"] == 1.0
    assert pd.isna(labeled.loc[2, "target_positive"])


def test_nn_confirmation_skips_without_enabling_torch_path():
    df = pd.DataFrame({"ret_1": [0.0, 0.01], "feature": [0.1, 0.2]})

    result = add_nn_confirmation(df, ["feature"], PipelineConfig(enable_nn_confirmation=False))

    assert "nn_probability_positive" in result
    assert (result["nn_signal"] == 0).all()
