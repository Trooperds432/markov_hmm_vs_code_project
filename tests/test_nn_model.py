from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.nn_model import (
    add_nn_confirmation,
    make_forward_labels,
    make_sequence_dataset,
    make_time_decay_weights,
    nn_forecasts_frame,
    nn_oos_split_info_frame,
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


def test_make_sequence_dataset_carries_sample_weights():
    matrix = pd.DataFrame({"a": range(6), "b": range(10, 16)}).to_numpy(dtype=float)
    labels = pd.Series([0, 1, 0, 1, float("nan"), 0]).to_numpy(dtype=float)
    sample_weights = pd.Series([1, 2, 3, 4, 5, 6]).to_numpy(dtype=float)

    dataset = make_sequence_dataset(
        matrix,
        labels,
        positions=pd.Index([2, 3, 4, 5]),
        lookback=3,
        sample_weights=sample_weights,
    )

    assert dataset.sample_weight.tolist() == [3.0, 4.0, 6.0]


def test_time_decay_weights_prioritise_recent_samples():
    weights = make_time_decay_weights(
        pd.Series([0, 50, 100]).to_numpy(),
        anchor_position=100,
        decay_factor=0.01,
        enabled=True,
    )

    assert weights[-1] > weights[0]
    assert round(float(weights.mean()), 6) == 1.0


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


def test_nn_oos_split_info_reports_structured_skip(monkeypatch):
    df = pd.DataFrame(
        {
            "ret_1": [0.0, 0.01, -0.01, 0.02],
            "feature": [0.1, 0.2, 0.3, 0.4],
            "hmm_window_id": [float("nan"), float("nan"), 1.0, 1.0],
        },
        index=pd.date_range("2020-01-01", periods=4),
    )
    cfg = PipelineConfig(enable_nn_confirmation=True)
    monkeypatch.setattr("src.markov_hmm_trader.nn_model.torch_available", lambda: False)

    add_nn_confirmation(df, ["feature"], cfg)
    split_info = nn_oos_split_info_frame(cfg)

    assert split_info.loc[0, "error"] == "torch_missing"
    assert split_info.loc[0, "strict_oos_enforced"] is True


def test_nn_forecasts_frame_uses_lstm_compatible_columns():
    df = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "nn_probability_positive": [0.7, 0.4],
            "nn_probability_std": [0.05, 0.02],
            "nn_adjusted_probability": [0.65, 0.38],
            "nn_signal": [1, -1],
            "target_positive": [1.0, 0.0],
            "nn_window_id": [1, 1],
            "nn_is_oos": [True, True],
            "nn_model_type": ["conv_lstm", "conv_lstm"],
            "nn_error": ["", ""],
        },
        index=pd.date_range("2020-01-01", periods=2),
    )

    forecasts = nn_forecasts_frame(df, PipelineConfig(nn_horizon_bars=20))

    assert {"Date", "Close", "Pred_Prob_1m", "Pred_Prob_Std_1m", "Adj_Prob_1m"}.issubset(
        forecasts.columns
    )
    assert forecasts.loc[0, "Pred_Prob_1m"] == 0.7
