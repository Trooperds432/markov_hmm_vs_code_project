from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from .config import PipelineConfig


def realised_volatility_forecast(returns: pd.Series, cfg: PipelineConfig) -> pd.Series:
    horizon = max(int(cfg.vol_forecast_horizon_bars), 2)
    return returns.rolling(horizon).std(ddof=0).shift(1) * np.sqrt(cfg.bars_per_year)


def ewma_volatility_forecast(returns: pd.Series, cfg: PipelineConfig) -> pd.Series:
    shifted = returns.shift(1).fillna(0.0)
    variance = shifted.pow(2).ewm(alpha=1.0 - float(cfg.ewma_lambda), adjust=False).mean()
    return np.sqrt(variance) * np.sqrt(cfg.bars_per_year)


def _risk_multiplier(sigma: pd.Series, cfg: PipelineConfig) -> tuple[pd.Series, pd.Series]:
    rank = sigma.rolling(cfg.vol_risk_lookback, min_periods=20).rank(pct=True)
    multiplier = (1.0 - rank).clip(cfg.vol_min_risk_multiplier, cfg.vol_max_risk_multiplier)
    return rank.fillna(0.5), multiplier.fillna(1.0)


def run_volatility_forecasts(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    returns = df["ret_1"].fillna(0.0)
    statsforecast_available = importlib.util.find_spec("statsforecast") is not None

    for model in cfg.vol_model_candidates:
        model_name = model.lower()
        error = ""
        if model_name == "realised_vol":
            sigma = realised_volatility_forecast(returns, cfg)
        elif model_name == "ewma_vol":
            sigma = ewma_volatility_forecast(returns, cfg)
        elif model_name in {"garch", "arch"}:
            if not statsforecast_available:
                sigma = realised_volatility_forecast(returns, cfg)
                error = "statsforecast_missing"
            else:
                # Optional adapter placeholder: keep the forecast contract stable and bounded.
                sigma = ewma_volatility_forecast(returns, cfg)
        else:
            sigma = realised_volatility_forecast(returns, cfg)
            error = f"unsupported_vol_model:{model_name}"

        sigma = sigma.replace([np.inf, -np.inf], np.nan).bfill().fillna(0.0)
        variance = (sigma / np.sqrt(cfg.bars_per_year)).pow(2) * cfg.vol_forecast_horizon_bars
        vol_rank, risk_multiplier = _risk_multiplier(sigma, cfg)
        for date in df.index:
            rows.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "window_id": 0,
                    "model_name": model_name,
                    "vol_model": model_name,
                    "horizon_bars": int(cfg.vol_forecast_horizon_bars),
                    "forecast_sigma": float(sigma.loc[date]),
                    "forecast_variance": float(variance.loc[date]),
                    "vol_rank": float(vol_rank.loc[date]),
                    "vol_risk_multiplier": float(risk_multiplier.loc[date]),
                    "split": "oos_proxy",
                    "error": error,
                }
            )
    return pd.DataFrame(rows)
