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
    higher_is_better: bool = True,
) -> dict[str, object]:
    train_validation_threshold, validation_test_threshold = _gap_thresholds(cfg)
    sign = 1.0 if higher_is_better else -1.0
    train_validation_gap = (
        float(sign * (train_metric - validation_metric))
        if np.isfinite(train_metric) and np.isfinite(validation_metric)
        else np.nan
    )
    validation_test_gap = (
        float(sign * (validation_metric - test_metric))
        if np.isfinite(validation_metric) and np.isfinite(test_metric)
        else np.nan
    )
    reasons = []
    if not (
        np.isfinite(train_metric) and np.isfinite(validation_metric) and np.isfinite(test_metric)
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
                higher_is_better=True,
            )
        )
    return pd.DataFrame(rows, columns=OVERFIT_COLUMNS)


def forecast_overfit_report(comparison: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    if comparison.empty:
        return pd.DataFrame(columns=OVERFIT_COLUMNS)
    for row in comparison.itertuples(index=False):
        validation_metric = float(
            getattr(row, "validation_mae", getattr(row, "cv_metric_value", np.nan))
        )
        test_metric = float(getattr(row, "test_mae", np.nan))
        rows.append(
            overfit_row(
                "forecast",
                getattr(row, "model_name", getattr(row, "candidate_id", "")),
                np.nan,
                validation_metric,
                test_metric,
                cfg,
                higher_is_better=False,
            )
        )
    return pd.DataFrame(rows, columns=OVERFIT_COLUMNS)


def volatility_overfit_report(forecasts: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    if forecasts.empty:
        return pd.DataFrame(columns=OVERFIT_COLUMNS)
    for model_name, group in forecasts.groupby("vol_model", dropna=False):
        errors = (
            group.get("abs_error", pd.Series(dtype=float))
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )
        if errors.empty:
            sigma = group.get("forecast_sigma", pd.Series(dtype=float)).replace(
                [np.inf, -np.inf],
                np.nan,
            )
            validation_metric = float(sigma.mean()) if sigma.notna().any() else np.nan
            test_metric = np.nan
        else:
            split = max(int(len(errors) * 0.7), 1)
            validation_metric = float(errors.iloc[:split].mean())
            test_metric = float(errors.iloc[split:].mean()) if len(errors) > split else np.nan
        rows.append(
            overfit_row(
                "volatility",
                model_name,
                np.nan,
                validation_metric,
                test_metric,
                cfg,
                higher_is_better=False,
            )
        )
    return pd.DataFrame(rows, columns=OVERFIT_COLUMNS)


def nn_overfit_report(training_metrics: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    if training_metrics.empty:
        return pd.DataFrame(columns=OVERFIT_COLUMNS)
    for fold_id, group in training_metrics.groupby("fold_id", dropna=False):
        train_losses = group.get("train_loss", pd.Series(dtype=float)).dropna()
        validation_losses = group.get("val_loss", pd.Series(dtype=float)).dropna()
        train_metric = float(train_losses.iloc[-1]) if not train_losses.empty else np.nan
        validation_metric = (
            float(validation_losses.iloc[-1]) if not validation_losses.empty else np.nan
        )
        rows.append(
            overfit_row(
                "nn",
                fold_id,
                train_metric,
                validation_metric,
                np.nan,
                cfg,
                higher_is_better=False,
            )
        )
    return pd.DataFrame(rows, columns=OVERFIT_COLUMNS)


def nested_overfit_report(candidate_results: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    if candidate_results.empty:
        return pd.DataFrame(columns=OVERFIT_COLUMNS)
    for row in candidate_results.itertuples(index=False):
        candidate_id = f"{getattr(row, 'outer_window_id', '')}:{getattr(row, 'candidate_id', '')}"
        rows.append(
            overfit_row(
                "nested",
                candidate_id,
                np.nan,
                float(getattr(row, "validation_score", np.nan)),
                np.nan,
                cfg,
                higher_is_better=True,
            )
        )
    return pd.DataFrame(rows, columns=OVERFIT_COLUMNS)


def append_overfit_reports(*frames: pd.DataFrame) -> pd.DataFrame:
    clean = [frame for frame in frames if frame is not None and not frame.empty]
    if not clean:
        return pd.DataFrame(columns=OVERFIT_COLUMNS)
    return pd.concat(clean, ignore_index=True).reindex(columns=OVERFIT_COLUMNS)
