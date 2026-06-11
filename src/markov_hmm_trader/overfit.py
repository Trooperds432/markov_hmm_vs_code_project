from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig

OVERFIT_COLUMNS = [
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


def _gap_thresholds(cfg: PipelineConfig) -> tuple[float, float]:
    train_validation = cfg.overfit_train_validation_gap_threshold
    validation_test = cfg.overfit_validation_test_gap_threshold
    if cfg.overfit_train_validation_gap_warn is not None:
        train_validation = float(cfg.overfit_train_validation_gap_warn)
    if cfg.overfit_validation_test_gap_warn is not None:
        validation_test = float(cfg.overfit_validation_test_gap_warn)
    return float(train_validation), float(validation_test)


def overfit_row(
    model_family: str,
    candidate_id: object,
    train_metric: float,
    validation_metric: float,
    test_metric: float,
    cfg: PipelineConfig,
) -> dict[str, object]:
    train_validation_threshold, validation_test_threshold = _gap_thresholds(cfg)
    train_validation_gap = (
        float(train_metric - validation_metric)
        if np.isfinite(train_metric) and np.isfinite(validation_metric)
        else np.nan
    )
    validation_test_gap = (
        float(validation_metric - test_metric)
        if np.isfinite(validation_metric) and np.isfinite(test_metric)
        else np.nan
    )
    reasons = []
    if not (
        np.isfinite(train_metric)
        and np.isfinite(validation_metric)
        and np.isfinite(test_metric)
    ):
        reasons.append("missing_metric")
    if np.isfinite(train_validation_gap) and train_validation_gap > train_validation_threshold:
        reasons.append("train_validation_gap_exceeds_threshold")
    if np.isfinite(validation_test_gap) and validation_test_gap > validation_test_threshold:
        reasons.append("validation_test_gap_exceeds_threshold")
    return {
        "model_family": model_family,
        "candidate_id": candidate_id,
        "train_metric": train_metric,
        "validation_metric": validation_metric,
        "test_metric": test_metric,
        "train_validation_gap": train_validation_gap,
        "validation_test_gap": validation_test_gap,
        "is_overfit_warning": bool(reasons),
        "reason": "|".join(reasons),
    }


def hmm_overfit_report(
    model_selection: pd.DataFrame,
    window_diagnostics: pd.DataFrame,
    cfg: PipelineConfig,
) -> pd.DataFrame:
    rows = []
    if model_selection.empty:
        return pd.DataFrame(columns=OVERFIT_COLUMNS)

    test_by_window = (
        window_diagnostics.set_index("window_id")["test_log_likelihood"].to_dict()
        if not window_diagnostics.empty and "test_log_likelihood" in window_diagnostics
        else {}
    )
    for row in model_selection.itertuples(index=False):
        rows.append(
            overfit_row(
                "hmm",
                getattr(row, "candidate_id", getattr(row, "seed", "")),
                float(getattr(row, "train_log_likelihood", np.nan)),
                float(getattr(row, "validation_log_likelihood", np.nan)),
                float(test_by_window.get(getattr(row, "window_id", None), np.nan)),
                cfg,
            )
        )
    return pd.DataFrame(rows, columns=OVERFIT_COLUMNS)


def append_overfit_reports(*frames: pd.DataFrame) -> pd.DataFrame:
    clean = [frame for frame in frames if frame is not None and not frame.empty]
    if not clean:
        return pd.DataFrame(columns=OVERFIT_COLUMNS)
    return pd.concat(clean, ignore_index=True).reindex(columns=OVERFIT_COLUMNS)
