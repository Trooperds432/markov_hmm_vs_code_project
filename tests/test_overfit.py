from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.model_overfit_report import build_model_overfit_report
from src.markov_hmm_trader.overfit import hmm_overfit_report, overfit_row


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
