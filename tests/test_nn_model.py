from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.nn_model import (
    add_nn_confirmation,
    make_forward_labels,
    make_sequence_dataset,
)


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


def test_make_sequence_dataset_respects_lookback_and_labels():
    matrix = pd.DataFrame({"a": range(6), "b": range(10, 16)}).to_numpy(dtype=float)
    labels = pd.Series([0, 1, 0, 1, float("nan"), 0]).to_numpy(dtype=float)

    dataset = make_sequence_dataset(matrix, labels, positions=pd.Index([2, 3, 4, 5]), lookback=3)

    assert dataset.X.shape == (3, 3, 2)
    assert dataset.positions.tolist() == [2, 3, 5]
    assert dataset.y.tolist() == [0.0, 1.0, 0.0]


def test_nn_confirmation_structured_skip_when_torch_missing(monkeypatch):
    df = pd.DataFrame(
        {
            "ret_1": [0.0, 0.01, -0.01, 0.02],
            "feature": [0.1, 0.2, 0.3, 0.4],
            "hmm_window_id": [float("nan"), float("nan"), 1.0, 1.0],
        }
    )
    monkeypatch.setattr("src.markov_hmm_trader.nn_model.torch_available", lambda: False)

    result = add_nn_confirmation(
        df,
        ["feature"],
        PipelineConfig(enable_nn_confirmation=True),
    )

    assert result["nn_error"].eq("torch_missing").all()
    assert result["nn_is_oos"].eq(False).all()
