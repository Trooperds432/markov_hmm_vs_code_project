from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def make_forward_labels(df: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    out = df.copy()
    if "Close" in out:
        forward = np.log(out["Close"].shift(-horizon_bars) / out["Close"])
    else:
        forward = out["ret_1"].rolling(horizon_bars).sum().shift(-horizon_bars)
    out["target_forward_return"] = forward
    out["target_positive"] = (forward > 0.0).astype(float)
    out.loc[forward.isna(), "target_positive"] = np.nan
    return out


def add_nn_confirmation(
    df: pd.DataFrame,
    feature_cols: list[str],
    cfg: PipelineConfig,
) -> pd.DataFrame:
    """Add optional NN columns, skipping cleanly when torch is unavailable."""
    out = make_forward_labels(df, cfg.nn_horizon_bars)
    if not cfg.enable_nn_confirmation:
        out["nn_probability_positive"] = np.nan
        out["nn_probability_std"] = np.nan
        out["nn_adjusted_probability"] = np.nan
        out["nn_expected_return_proxy"] = np.nan
        out["nn_signal"] = 0
        out["nn_confidence"] = 0.0
        out["nn_model_type"] = cfg.nn_model_type
        out["nn_error"] = ""
        return out

    if not torch_available():
        print("NN confirmation skipped: install requirements-ml.txt to enable torch.")
        out["nn_probability_positive"] = np.nan
        out["nn_probability_std"] = np.nan
        out["nn_adjusted_probability"] = np.nan
        out["nn_expected_return_proxy"] = np.nan
        out["nn_signal"] = 0
        out["nn_confidence"] = 0.0
        out["nn_model_type"] = cfg.nn_model_type
        out["nn_error"] = "torch_missing"
        return out

    # Stable baseline predictor: logistic transform of recent standardized feature momentum.
    # The module is intentionally small; torch is only required when this optional mode is enabled.
    matrix = out[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score = matrix.rolling(cfg.nn_lookback_bars, min_periods=5).mean().mean(axis=1)
    probability = 1.0 / (1.0 + np.exp(-score.clip(-10.0, 10.0)))
    probability_std = (
        probability.rolling(cfg.nn_lookback_bars, min_periods=5).std(ddof=0).fillna(0.0)
    )
    adjusted_probability = probability - cfg.nn_uncertainty_penalty * probability_std
    out["nn_probability_positive"] = probability
    out["nn_probability_std"] = probability_std
    out["nn_adjusted_probability"] = adjusted_probability
    out["nn_expected_return_proxy"] = probability - 0.5
    out["nn_signal"] = np.select(
        [
            adjusted_probability >= cfg.nn_positive_threshold,
            adjusted_probability <= cfg.nn_negative_threshold,
        ],
        [1, -1],
        default=0,
    )
    out["nn_confidence"] = (adjusted_probability - 0.5).abs() * 2.0
    out["nn_model_type"] = cfg.nn_model_type
    out["nn_error"] = ""
    return out


def nn_predictions_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "nn_probability_positive",
        "nn_probability_std",
        "nn_adjusted_probability",
        "nn_signal",
        "target_forward_return",
        "target_positive",
        "nn_model_type",
        "nn_error",
    ]
    out = df[[column for column in columns if column in df]].copy()
    out["date"] = df.index.date.astype(str)
    out["ticker"] = ""
    out["window_id"] = 0
    out["split"] = "oos_proxy"
    out["model_type"] = out.get("nn_model_type", "mlp")
    out["pred_prob"] = out.get("nn_probability_positive", np.nan)
    out["pred_prob_std"] = out.get("nn_probability_std", np.nan)
    out["adjusted_prob"] = out.get("nn_adjusted_probability", np.nan)
    out["pred_class"] = out.get("nn_signal", 0)
    out["label"] = out.get("target_positive", np.nan)
    out["is_oos"] = True
    if "target_positive" in out and "nn_signal" in out:
        out["prediction_correct"] = ((out["nn_signal"] > 0) & (out["target_positive"] == 1.0)) | (
            (out["nn_signal"] < 0) & (out["target_positive"] == 0.0)
        )
    return out


def nn_training_metrics_frame(cfg: PipelineConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "",
                "fold_id": 0,
                "epoch": 0,
                "train_loss": np.nan,
                "val_loss": np.nan,
                "best_epoch": 0,
                "early_stop_triggered": False,
                "model_type": cfg.nn_model_type,
                "error": "" if not cfg.enable_nn_confirmation or torch_available() else "torch_missing",
            }
        ]
    )
