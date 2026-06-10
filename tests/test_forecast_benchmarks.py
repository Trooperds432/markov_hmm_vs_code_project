from __future__ import annotations

import sys
from types import ModuleType

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


def test_forecast_benchmarks_use_statsforecast_cross_validation(monkeypatch):
    calls = {"cross_validation": 0}

    class FakeModel:
        def __init__(self, season_length=1, alias="model"):
            self.alias = alias

        def __repr__(self):
            return self.alias

    class FakeStatsForecast:
        def __init__(self, models, freq, n_jobs=1, fallback_model=None):
            self.models = models
            self.freq = freq
            self.fallback_model = fallback_model

        def cross_validation(
            self,
            df,
            h,
            n_windows,
            step_size,
            level,
            refit,
            id_col,
            time_col,
            target_col,
        ):
            calls["cross_validation"] += 1
            model_name = repr(self.models[0])
            rows = df.tail(h * n_windows).copy()
            rows["cutoff"] = rows["ds"].min()
            rows[model_name] = rows["y"] + 0.001
            rows[f"{model_name}-lo-80"] = rows[model_name] - 0.01
            rows[f"{model_name}-hi-80"] = rows[model_name] + 0.01
            rows[f"{model_name}-lo-95"] = rows[model_name] - 0.02
            rows[f"{model_name}-hi-95"] = rows[model_name] + 0.02
            return rows

    statsforecast_module = ModuleType("statsforecast")
    statsforecast_module.StatsForecast = FakeStatsForecast
    models_module = ModuleType("statsforecast.models")
    models_module.Naive = FakeModel
    models_module.AutoARIMA = FakeModel
    models_module.AutoETS = FakeModel
    models_module.AutoTheta = FakeModel
    monkeypatch.setitem(sys.modules, "statsforecast", statsforecast_module)
    monkeypatch.setitem(sys.modules, "statsforecast.models", models_module)

    dates = pd.bdate_range("2020-01-01", periods=80)
    df = pd.DataFrame(
        {
            "ret_1": np.random.default_rng(5).normal(0.0, 0.01, len(dates)),
            "hmm_signal": 0.0,
        },
        index=dates,
    )
    cfg = PipelineConfig(
        forecast_models=("auto_arima",),
        forecast_horizon_bars=5,
        forecast_cv_windows=2,
    )

    outputs = run_forecast_benchmarks(df, cfg)

    assert calls["cross_validation"] == 1
    assert outputs["forecast_predictions"]["split"].eq("statsforecast_cv").all()
    assert outputs["forecast_model_comparison"].loc[0, "error"] == ""
