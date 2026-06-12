from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..states import STATE_BEAR, STATE_BULL


def glassdoor_position_multiplier(row: pd.Series, cfg: PipelineConfig) -> float:
    stress = float(row.get("glassdoor_stress_z", np.nan))
    if not np.isfinite(stress):
        return 1.0
    if stress > 2.5:
        return float(cfg.glassdoor_stress_position_mult_extreme)
    if stress > 1.5:
        return float(cfg.glassdoor_stress_position_mult_moderate)
    return 1.0


def wsb_position_multiplier(row: pd.Series, cfg: PipelineConfig) -> float:
    z = float(row.get("wsb_mentions_z_20d", np.nan))
    if not np.isfinite(z):
        return 1.0
    if z > float(cfg.wsb_extreme_attention_z):
        return float(cfg.wsb_attention_position_mult_extreme)
    if z > float(cfg.wsb_high_attention_z):
        return float(cfg.wsb_attention_position_mult_high)
    return 1.0


def valuation_position_multiplier(row: pd.Series, cfg: PipelineConfig) -> float:
    mos = float(row.get("margin_of_safety", np.nan))
    bull = float(row.get(STATE_BULL, np.nan))
    bear = float(row.get(STATE_BEAR, np.nan))
    if not np.isfinite(mos):
        return 1.0
    if bull > bear and mos > float(cfg.valuation_margin_threshold):
        return float(cfg.valuation_bull_boost_mult)
    return 1.0


def _positive_finite(value: object) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(numeric):
        return 0.0
    return max(numeric, 0.0)


def compute_alt_data_overlay(row: pd.Series, cfg: PipelineConfig) -> dict[str, float | bool]:
    glassdoor_mult = glassdoor_position_multiplier(row, cfg)
    wsb_mult = wsb_position_multiplier(row, cfg)
    valuation_mult = valuation_position_multiplier(row, cfg)
    combined = float(np.clip(glassdoor_mult * wsb_mult * valuation_mult, 0.25, 1.50))
    risk = float(
        np.mean(
            [
                _positive_finite(row.get("glassdoor_stress_z", 0.0)),
                _positive_finite(row.get("wsb_mentions_z_20d", 0.0)),
                _positive_finite(-float(row.get("margin_of_safety", 0.0) or 0.0)),
            ]
        )
    )
    return {
        "glassdoor_position_multiplier": glassdoor_mult,
        "wsb_position_multiplier": wsb_mult,
        "valuation_position_multiplier": valuation_mult,
        "alt_data_position_multiplier": combined,
        "alt_data_risk_score": risk,
        "alt_data_confidence_score": float(np.clip(1.0 / (1.0 + risk), 0.0, 1.0)),
        "alt_data_warning_flag": not np.isfinite(combined),
    }


def adjusted_portfolio_score(row: pd.Series, cfg: PipelineConfig) -> float:
    hmm_score = float(row.get(STATE_BULL, row.get("bull_probability", 0.0))) - float(
        row.get(STATE_BEAR, row.get("bear_probability", 0.0))
    )
    stress = max(float(row.get("glassdoor_stress_z", 0.0) or 0.0) - 1.5, 0.0)
    wsb = max(float(row.get("wsb_mentions_z_20d", 0.0) or 0.0) - 3.0, 0.0)
    mos = float(np.clip(row.get("margin_of_safety", 0.0) or 0.0, -0.5, 0.5))
    valuation_boost = 0.15 * mos if hmm_score > 0 else min(0.0, 0.15 * mos)
    return float(hmm_score - 0.10 * stress - 0.10 * wsb + valuation_boost)
