from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def apply_statistical_confidence_scaling(row: pd.Series, cfg: PipelineConfig) -> float:
    """Return an exposure multiplier from trailing statistical reliability diagnostics."""
    if not cfg.use_statistical_confidence_scaling:
        return 1.0

    mult = 1.0
    if float(row.get("stationarity_risk_score", 0.0) or 0.0) > 1.5:
        mult *= 0.75
    if float(row.get("parameter_instability_z", 0.0) or 0.0) > 1.5:
        mult *= 0.75
    if float(row.get("market_model_r2", 0.0) or 0.0) > 0.70:
        mult *= 0.80
    if float(row.get("pc1_explained_variance", 0.0) or 0.0) > 0.55:
        mult *= 0.80
    if float(row.get("systemic_correlation_score", 0.0) or 0.0) > 1.5:
        mult *= 0.75
    if float(row.get("residual_tail_risk_score", 0.0) or 0.0) > 1.5:
        mult *= 0.75

    return float(
        np.clip(
            mult,
            float(cfg.min_statistical_confidence_mult),
            float(cfg.max_statistical_confidence_mult),
        )
    )
