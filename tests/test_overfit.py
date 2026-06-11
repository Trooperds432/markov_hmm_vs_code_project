from __future__ import annotations

import pandas as pd
import pytest

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.model_overfit_report import build_model_overfit_report
from src.markov_hmm_trader.overfit import (
    forecast_overfit_report,
    hmm_overfit_report,
    nested_overfit_report,
    nn_overfit_report,
    overfit_row,
    volatility_overfit_report,
)


def test_overfit_row_flags_large_metric_gaps():
    cfg = PipelineConfig(
        overfit_train_validation_gap_threshold=0.5,
        overfit_validation_test_gap_threshold=0.25,
    )

    row = overfit_row("hmm", 1, train_metric=2.0, validation_metric=1.0, test_metric=0.5, cfg=cfg)

    assert row["is_overfit_warning"] is True
    assert "train_validation_gap_exceeds_threshold" in row["reason"]
    assert "validation_test_gap_exceeds_threshold" in row["reason"]


def test_overfit_row_flags_missing_metric():
    row = overfit_row(
        "hmm",
        1,
        train_metric=2.0,
        validation_metric=float("nan"),
        test_metric=0.5,
        cfg=PipelineConfig(),
    )

    assert row["is_overfit_warning"] is True
    assert row["reason"] == "missing_metric"


def test_overfit_row_supports_lower_is_better_metrics():
    cfg = PipelineConfig(
        overfit_train_validation_gap_threshold=0.25,
        overfit_validation_test_gap_threshold=0.25,
    )

    row = overfit_row(
        "nn",
        "fold-1",
        train_metric=0.2,
        validation_metric=0.6,
        test_metric=0.9,
        cfg=cfg,
        higher_is_better=False,
    )

    assert row["train_validation_gap"] == pytest.approx(0.4)
    assert row["validation_test_gap"] == pytest.approx(0.3)
    assert "train_validation_gap_exceeds_threshold" in row["reason"]
    assert "validation_test_gap_exceeds_threshold" in row["reason"]


def test_hmm_overfit_report_uses_candidate_rows():
    cfg = PipelineConfig()
    candidates = pd.DataFrame(
        [
            {
                "window_id": 1,
                "candidate_id": 7,
                "train_log_likelihood": 10.0,
                "validation_log_likelihood": 8.0,
            }
        ]
    )
    windows = pd.DataFrame([{"window_id": 1, "test_log_likelihood": 7.0}])

    report = hmm_overfit_report(candidates, windows, cfg)

    assert report.loc[0, "model_family"] == "hmm"
    assert report.loc[0, "candidate_id"] == 7
    assert report.loc[0, "train_validation_gap"] == 2.0


def test_optional_family_overfit_reports_create_structured_rows():
    cfg = PipelineConfig()
    forecast = forecast_overfit_report(
        pd.DataFrame([{"model_name": "auto_arima", "cv_metric_value": 0.12}]),
        cfg,
    )
    volatility = volatility_overfit_report(
        pd.DataFrame(
            [
                {"vol_model": "ewma_vol", "forecast_sigma": 0.2},
                {"vol_model": "ewma_vol", "forecast_sigma": 0.4},
            ]
        ),
        cfg,
    )
    nn = nn_overfit_report(
        pd.DataFrame(
            [
                {"fold_id": 1, "epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
                {"fold_id": 1, "epoch": 2, "train_loss": 0.4, "val_loss": 0.7},
            ]
        ),
        cfg,
    )
    nested = nested_overfit_report(
        pd.DataFrame([{"outer_window_id": 2, "candidate_id": 3, "validation_score": 1.5}]),
        cfg,
    )

    report = pd.concat([forecast, volatility, nn, nested], ignore_index=True)

    assert set(report["model_family"]) == {"forecast", "volatility", "nn", "nested"}
    assert all("missing_metric" in reason for reason in report["reason"])
    assert "train_validation_gap_exceeds_threshold" in nn.loc[0, "reason"]
    assert volatility.loc[0, "validation_metric"] == pytest.approx(0.3)
    assert nn.loc[0, "train_metric"] == 0.4
    assert nn.loc[0, "validation_metric"] == 0.7
    assert nested.loc[0, "candidate_id"] == "2:3"


def test_model_overfit_report_wrapper_preserves_columns():
    report = build_model_overfit_report(
        pd.DataFrame(),
        pd.DataFrame(),
        PipelineConfig(),
    )

    assert list(report.columns) == [
        "model_family",
        "candidate_id",
        "train_metric",
        "validation_metric",
        "test_metric",
        "train_validation_gap",
        "validation_test_gap",
        "is_overfit_warning",
        "reason",
    ]
