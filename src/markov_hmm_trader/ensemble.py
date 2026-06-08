from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .states import STATE_BEAR, STATE_BULL, STATE_SIDE


def ensemble_signal_for_row(row: pd.Series, cfg: PipelineConfig) -> float:
    hmm_signal = float(row.get(STATE_BULL, 0.0) - row.get(STATE_BEAR, 0.0))
    nn_probability = row.get("nn_probability_positive", np.nan)
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
    return hmm_signal


def add_ensemble_signals(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    if cfg.ensemble_mode == "hmm_only":
        out["ensemble_signal"] = out.get("hmm_signal", out[STATE_BULL] - out[STATE_BEAR])
    else:
        out["ensemble_signal"] = out.apply(lambda row: ensemble_signal_for_row(row, cfg), axis=1)
    out["hmm_signal"] = out.get("hmm_signal", out[STATE_BULL] - out[STATE_BEAR])
    out["agreement"] = np.sign(out["hmm_signal"]) == np.sign(out.get("nn_signal", 0))
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
    columns = [
        "hmm_signal",
        "nn_signal",
        "ensemble_signal",
        "agreement",
        "position",
    ]
    return df[[column for column in columns if column in df]].copy()
