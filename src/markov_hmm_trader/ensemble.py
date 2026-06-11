from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .states import STATE_BEAR, STATE_BULL, STATE_SIDE


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if np.isfinite(numeric) else default


def ensemble_signal_for_row(row: pd.Series, cfg: PipelineConfig) -> float:
    hmm_signal = float(row.get(STATE_BULL, 0.0) - row.get(STATE_BEAR, 0.0))
    nn_probability = row.get(
        "nn_adjusted_probability",
        row.get("nn_probability_positive", np.nan),
    )
    nn_signal = 0.0
    if np.isfinite(nn_probability):
        if nn_probability >= cfg.nn_positive_threshold:
            nn_signal = 1.0
        elif nn_probability <= cfg.nn_negative_threshold:
            nn_signal = -1.0

    if cfg.ensemble_mode == "nn_only":
        return nn_signal
    if cfg.ensemble_mode == "hmm_and_nn_agree":
        if hmm_signal > 0.0 and nn_signal > 0.0:
            return min(1.0, hmm_signal)
        if hmm_signal < 0.0 and nn_signal < 0.0:
            return max(-1.0, hmm_signal)
        return 0.0
    if cfg.ensemble_mode == "weighted_blend":
        return float(np.clip(cfg.hmm_weight * hmm_signal + cfg.nn_weight * nn_signal, -1.0, 1.0))
    if cfg.ensemble_mode == "multi_signal":
        observable_signal = _finite_float(row.get("observable_markov_signal", 0.0))
        forecast_signal = _finite_float(row.get("forecast_signal", 0.0))
        risk_multiplier = _finite_float(
            row.get("vol_risk_multiplier", row.get("risk_multiplier", 1.0)),
            default=1.0,
        )
        risk_multiplier = float(np.clip(risk_multiplier, 0.0, max(cfg.max_vol_mult, 1.0)))
        raw_signal = (
            cfg.hmm_weight * hmm_signal
            + cfg.nn_weight * nn_signal
            + cfg.observable_markov_weight * observable_signal
            + cfg.forecast_weight * forecast_signal
        )
        return float(np.clip(raw_signal * risk_multiplier, -1.0, 1.0))
    return hmm_signal


def add_ensemble_signals(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    if cfg.ensemble_mode == "hmm_only":
        out["ensemble_signal"] = out.get("hmm_signal", out[STATE_BULL] - out[STATE_BEAR])
    else:
        out["ensemble_signal"] = out.apply(lambda row: ensemble_signal_for_row(row, cfg), axis=1)
    out["hmm_signal"] = out.get("hmm_signal", out[STATE_BULL] - out[STATE_BEAR])
    out["observable_markov_signal"] = out.get("observable_markov_signal", 0.0)
    out["forecast_signal"] = out.get("forecast_signal", 0.0)
    out["risk_multiplier"] = out.get("vol_risk_multiplier", 1.0)
    out["agreement"] = np.sign(out["hmm_signal"]) == np.sign(out.get("nn_signal", 0))
    out["signal_direction"] = np.sign(out["ensemble_signal"])
    out["signal_conviction"] = out["ensemble_signal"].abs()
    out["ensemble_regime"] = np.select(
        [
            out[STATE_BULL] >= out[[STATE_BEAR, STATE_SIDE]].max(axis=1),
            out[STATE_BEAR] >= out[[STATE_BULL, STATE_SIDE]].max(axis=1),
        ],
        [STATE_BULL, STATE_BEAR],
        default=STATE_SIDE,
    )
    return out


def ensemble_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["date"] = df.index.date.astype(str)
    out["hmm_signal"] = df.get("hmm_signal", 0.0)
    out["observable_markov_signal"] = df.get("observable_markov_signal", 0.0)
    out["forecast_signal"] = df.get("forecast_signal", 0.0)
    out["nn_signal"] = df.get("nn_signal", 0)
    out["ensemble_signal"] = df.get("ensemble_signal", df.get("hmm_signal", 0.0))
    out["signal_direction"] = np.sign(out["ensemble_signal"])
    out["signal_conviction"] = out["ensemble_signal"].abs()
    out["risk_multiplier"] = df.get("vol_risk_multiplier", 1.0)
    out["final_position"] = df.get("position", 0.0)
    out["hmm_agree_nn"] = np.sign(out["hmm_signal"]) == np.sign(out["nn_signal"])
    out["forecast_confirm_flag"] = np.sign(out["hmm_signal"]) == np.sign(out["forecast_signal"])
    out["stationary_bear_gate"] = False
    out["garch_gate"] = out["risk_multiplier"] < 1.0
    out["final_trade_allowed"] = out["final_position"].abs() > 0.0
    return out
