from __future__ import annotations

import pandas as pd

from .config import PipelineConfig
from .overfit import (
    OVERFIT_COLUMNS,
    append_overfit_reports,
    forecast_overfit_report,
    hmm_overfit_report,
    nested_overfit_report,
    nn_overfit_report,
    overfit_row,
    volatility_overfit_report,
)


def build_model_overfit_report(
    hmm_model_selection: pd.DataFrame,
    window_diagnostics: pd.DataFrame,
    cfg: PipelineConfig,
    *additional_reports: pd.DataFrame,
) -> pd.DataFrame:
    """Build the additive model overfit report used by Stage 2 diagnostics."""
    frames = [hmm_overfit_report(hmm_model_selection, window_diagnostics, cfg)]
    frames.extend(additional_reports)
    report = append_overfit_reports(*frames)
    if report.empty:
        return pd.DataFrame(columns=OVERFIT_COLUMNS)
    return report.reindex(columns=OVERFIT_COLUMNS)


__all__ = [
    "OVERFIT_COLUMNS",
    "append_overfit_reports",
    "build_model_overfit_report",
    "forecast_overfit_report",
    "hmm_overfit_report",
    "nested_overfit_report",
    "nn_overfit_report",
    "overfit_row",
    "volatility_overfit_report",
]
