from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.forecast_benchmarks import run_forecast_benchmarks


def test_forecast_benchmarks_emit_required_columns():
    dates = pd.bdate_range("2020-01-01", periods=80)
    df = pd.DataFrame(
        {
            "ret_1": np.random.default_rng(4).normal(0.0, 0.01, len(dates)),
            "hmm_signal": 0.0,
        },
        index=dates,
    )
    cfg = PipelineConfig(forecast_models=("naive", "auto_arima"), forecast_horizon_bars=5)

    outputs = run_forecast_benchmarks(df, cfg)
    predictions = outputs["forecast_predictions"]
    comparison = outputs["forecast_model_comparison"]

    assert {
        "date",
        "model_name",
        "horizon_bars",
        "forecast_mean",
        "lower_80",
        "upper_95",
        "direction_signal",
        "error",
    }.issubset(predictions.columns)
    assert {"model_name", "cv_metric_name", "cv_metric_value", "selected", "error"}.issubset(
        comparison.columns
    )
