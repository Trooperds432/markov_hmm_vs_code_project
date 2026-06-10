from __future__ import annotations

from importlib import import_module

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


def _horizon_variance_from_annual_sigma(sigma: pd.Series, cfg: PipelineConfig) -> pd.Series:
    return (sigma / np.sqrt(cfg.bars_per_year)).pow(2) * cfg.vol_forecast_horizon_bars


def _risk_multiplier(sigma: pd.Series, cfg: PipelineConfig) -> tuple[pd.Series, pd.Series]:
    rank = sigma.rolling(cfg.vol_risk_lookback, min_periods=20).rank(pct=True)
    multiplier = (1.0 - rank).clip(cfg.vol_min_risk_multiplier, cfg.vol_max_risk_multiplier)
    return rank.fillna(0.5), multiplier.fillna(1.0)


def _try_import_vol_model(model_name: str) -> tuple[type | None, str]:
    try:
        models = import_module("statsforecast.models")
        if model_name == "garch":
            return models.GARCH, ""
        if model_name == "arch":
            return models.ARCH, ""
    except Exception as err:
        return None, f"statsforecast_missing:{type(err).__name__}"
    return None, f"unsupported_vol_model:{model_name}"


def _statsforecast_volatility_forecast(
    returns: pd.Series,
    cfg: PipelineConfig,
    model_name: str,
) -> tuple[pd.Series, pd.Series, str]:
    model_cls, import_error = _try_import_vol_model(model_name)
    if model_cls is None:
        sigma = realised_volatility_forecast(returns, cfg)
        return sigma, _horizon_variance_from_annual_sigma(sigma, cfg), import_error

    horizon = max(int(cfg.vol_forecast_horizon_bars), 1)
    refit_frequency = max(int(cfg.vol_forecast_refit_frequency), 1)
    min_obs = max(30, horizon + max(int(cfg.garch_p), int(cfg.garch_q), int(cfg.arch_p)) + 5)
    annual_sigma = pd.Series(np.nan, index=returns.index, dtype=float)
    horizon_variance = pd.Series(np.nan, index=returns.index, dtype=float)
    errors = []

    for position in range(min_obs, len(returns), refit_frequency):
        train = returns.iloc[:position].replace([np.inf, -np.inf], np.nan).dropna()
        if len(train) < min_obs:
            continue
        try:
            if model_name == "garch":
                model = model_cls(p=int(cfg.garch_p), q=int(cfg.garch_q))
            else:
                model = model_cls(p=int(cfg.arch_p))
            forecast = model.forecast(y=train.to_numpy(dtype=float), h=horizon)
            sigma2 = np.asarray(forecast.get("sigma2"), dtype=float)
            sigma2 = sigma2[np.isfinite(sigma2)]
            if sigma2.size == 0:
                raise ValueError("empty_sigma2")
            per_bar_variance = float(np.mean(np.maximum(sigma2, 0.0)))
            forecast_horizon_variance = float(np.sum(np.maximum(sigma2, 0.0)))
            value = np.sqrt(per_bar_variance) * np.sqrt(cfg.bars_per_year)
            end = min(position + refit_frequency, len(returns))
            annual_sigma.iloc[position:end] = value
            horizon_variance.iloc[position:end] = forecast_horizon_variance
        except Exception as err:
            errors.append(type(err).__name__)

    fallback_sigma = ewma_volatility_forecast(returns, cfg)
    annual_sigma = annual_sigma.ffill().fillna(fallback_sigma)
    horizon_variance = horizon_variance.ffill().fillna(
        _horizon_variance_from_annual_sigma(fallback_sigma, cfg)
    )
    if annual_sigma.notna().sum() == 0:
        return (
            fallback_sigma,
            _horizon_variance_from_annual_sigma(fallback_sigma, cfg),
            (f"statsforecast_failed:{errors[-1]}" if errors else "statsforecast_no_forecasts"),
        )
    if errors:
        return annual_sigma, horizon_variance, f"partial_statsforecast_fallback:{errors[-1]}"
    return annual_sigma, horizon_variance, ""


def run_volatility_forecasts(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    rows = []
    returns = df["ret_1"].fillna(0.0)

    for model in cfg.vol_model_candidates:
        model_name = model.lower()
        error = ""
        if model_name == "realised_vol":
            sigma = realised_volatility_forecast(returns, cfg)
            variance = _horizon_variance_from_annual_sigma(sigma, cfg)
        elif model_name == "ewma_vol":
            sigma = ewma_volatility_forecast(returns, cfg)
            variance = _horizon_variance_from_annual_sigma(sigma, cfg)
        elif model_name in {"garch", "arch"}:
            sigma, variance, error = _statsforecast_volatility_forecast(
                returns,
                cfg,
                model_name,
            )
        else:
            sigma = realised_volatility_forecast(returns, cfg)
            variance = _horizon_variance_from_annual_sigma(sigma, cfg)
            error = f"unsupported_vol_model:{model_name}"

        sigma = sigma.replace([np.inf, -np.inf], np.nan).bfill().fillna(0.0)
        variance = variance.replace([np.inf, -np.inf], np.nan).bfill().fillna(0.0)
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
                    "forecast_variance": float(max(variance.loc[date], 0.0)),
                    "vol_rank": float(vol_rank.loc[date]),
                    "vol_risk_multiplier": float(risk_multiplier.loc[date]),
                    "split": "oos_proxy",
                    "error": error,
                }
            )
    return pd.DataFrame(rows)
