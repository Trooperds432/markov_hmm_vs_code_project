from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig


@dataclass
class StatsForecastImports:
    statsforecast: type
    naive: type
    auto_arima: type
    auto_ets: type
    auto_theta: type


def _try_import_statsforecast() -> tuple[StatsForecastImports | None, str]:
    try:
        core = import_module("statsforecast")
        models = import_module("statsforecast.models")
        return (
            StatsForecastImports(
                statsforecast=getattr(core, "StatsForecast"),
                naive=getattr(models, "Naive"),
                auto_arima=getattr(models, "AutoARIMA"),
                auto_ets=getattr(models, "AutoETS"),
                auto_theta=getattr(models, "AutoTheta"),
            ),
            "",
        )
    except Exception as err:
        return None, f"statsforecast_missing:{type(err).__name__}"


def _instantiate(model_cls: type, **kwargs: Any) -> Any:
    signature = inspect.signature(model_cls)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return model_cls(**accepted)


def _statsforecast_model(model_name: str, imports: StatsForecastImports, cfg: PipelineConfig) -> Any:
    normalized = model_name.lower()
    alias = normalized
    if normalized == "naive":
        return _instantiate(imports.naive, alias=alias)
    if normalized == "auto_arima":
        return _instantiate(imports.auto_arima, season_length=1, alias=alias)
    if normalized == "auto_ets":
        return _instantiate(imports.auto_ets, season_length=1, alias=alias)
    if normalized == "auto_theta":
        return _instantiate(imports.auto_theta, season_length=1, alias=alias)
    raise ValueError(f"unsupported_forecast_model:{normalized}")


def _naive_forecast(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    model_name: str,
    error: str = "",
) -> pd.DataFrame:
    horizon = int(cfg.forecast_horizon_bars)
    mean = df["ret_1"].rolling(max(horizon, 2)).mean().shift(1).fillna(0.0) * horizon
    sigma = df["ret_1"].rolling(max(horizon, 2)).std(ddof=0).shift(1).fillna(0.0) * np.sqrt(
        horizon
    )
    return pd.DataFrame(
        {
            "date": df.index.date.astype(str),
            "window_id": 0,
            "model_name": model_name,
            "horizon_bars": horizon,
            "forecast_mean": mean.values,
            "lower_80": (mean - 1.2816 * sigma).values,
            "upper_80": (mean + 1.2816 * sigma).values,
            "lower_95": (mean - 1.96 * sigma).values,
            "upper_95": (mean + 1.96 * sigma).values,
            "direction_signal": np.sign(mean).astype(float).values,
            "split": "oos_proxy",
            "error": error,
        },
        index=df.index,
    )


def _statsforecast_frame(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": str(cfg.ticker),
            "ds": pd.to_datetime(df.index),
            "y": df["ret_1"].fillna(0.0).astype(float).values,
        }
    )


def _forecast_columns(cv: pd.DataFrame) -> list[str]:
    reserved = {"unique_id", "ds", "cutoff", "y"}
    return [column for column in cv.columns if column not in reserved]


def _column_with_suffix(columns: list[str], suffix: str) -> str | None:
    matches = [column for column in columns if column.endswith(suffix)]
    return matches[0] if matches else None


def _cross_validation_forecast(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    model_name: str,
    imports: StatsForecastImports,
) -> tuple[pd.DataFrame, float, str]:
    horizon = int(cfg.forecast_horizon_bars)
    step = max(int(cfg.forecast_step_size), 1)
    max_windows = max((len(df) - horizon) // step, 0)
    n_windows = min(max(int(cfg.forecast_cv_windows), 1), max_windows)
    if n_windows < 1:
        return (
            _naive_forecast(df, cfg, model_name, error="statsforecast_insufficient_history"),
            np.nan,
            "statsforecast_insufficient_history",
        )

    model = _statsforecast_model(model_name, imports, cfg)
    fallback = _instantiate(imports.naive, alias=f"{model_name}_fallback")
    sf = imports.statsforecast(
        models=[model],
        freq=cfg.forecast_freq,
        n_jobs=1,
        fallback_model=fallback,
    )
    levels = list(cfg.forecast_levels)
    cv = sf.cross_validation(
        df=_statsforecast_frame(df, cfg),
        h=horizon,
        n_windows=n_windows,
        step_size=step,
        level=levels,
        refit=cfg.forecast_refit_every,
        id_col="unique_id",
        time_col="ds",
        target_col="y",
    )
    forecast_columns = _forecast_columns(cv)
    mean_candidates = [
        column for column in forecast_columns if "-lo-" not in column and "-hi-" not in column
    ]
    if not mean_candidates:
        raise ValueError("statsforecast_no_mean_column")
    mean_col = mean_candidates[0]
    lower_80 = _column_with_suffix(forecast_columns, "-lo-80")
    upper_80 = _column_with_suffix(forecast_columns, "-hi-80")
    lower_95 = _column_with_suffix(forecast_columns, "-lo-95")
    upper_95 = _column_with_suffix(forecast_columns, "-hi-95")

    cv = cv.copy()
    cv["window_id"] = pd.factorize(cv.get("cutoff", pd.Series(0, index=cv.index)))[0] + 1
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(cv["ds"]).dt.date.astype(str),
            "window_id": cv["window_id"].astype(int),
            "model_name": model_name,
            "horizon_bars": horizon,
            "forecast_mean": cv[mean_col].astype(float),
            "lower_80": cv[lower_80].astype(float) if lower_80 else np.nan,
            "upper_80": cv[upper_80].astype(float) if upper_80 else np.nan,
            "lower_95": cv[lower_95].astype(float) if lower_95 else np.nan,
            "upper_95": cv[upper_95].astype(float) if upper_95 else np.nan,
            "direction_signal": np.sign(cv[mean_col].astype(float)),
            "split": "statsforecast_cv",
            "error": "",
        }
    )
    metric = float((cv[mean_col].astype(float) - cv["y"].astype(float)).abs().mean())
    return predictions, metric, ""


def _metric_against_realised(
    df: pd.DataFrame,
    predictions: pd.DataFrame,
    cfg: PipelineConfig,
) -> float:
    realised = (
        df["ret_1"].rolling(cfg.forecast_horizon_bars).sum().shift(-cfg.forecast_horizon_bars)
    )
    pred = predictions.copy()
    pred.index = pd.to_datetime(pred["date"])
    aligned = pred["forecast_mean"].groupby(level=0).last().reindex(df.index)
    metric = (aligned - realised).abs().mean()
    return float(metric) if np.isfinite(metric) else np.nan


def run_forecast_benchmarks(df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    prediction_frames = []
    comparison_rows = []
    imports, import_error = _try_import_statsforecast()

    for model_name in cfg.forecast_models:
        normalized = model_name.lower()
        error = ""
        if imports is None and normalized == "naive":
            predictions = _naive_forecast(df, cfg, normalized)
            metric = _metric_against_realised(df, predictions, cfg)
        elif imports is None:
            error = import_error or "statsforecast_missing"
            predictions = _naive_forecast(df, cfg, normalized, error=error)
            metric = _metric_against_realised(df, predictions, cfg)
        else:
            try:
                predictions, metric, error = _cross_validation_forecast(
                    df, cfg, normalized, imports
                )
            except Exception as err:
                error = f"statsforecast_failed:{type(err).__name__}"
                predictions = _naive_forecast(df, cfg, normalized, error=error)
                metric = _metric_against_realised(df, predictions, cfg)

        prediction_frames.append(predictions)
        comparison_rows.append(
            {
                "window_id": 0,
                "model_name": normalized,
                "cv_metric_name": "mae",
                "cv_metric_value": metric,
                "selected": False,
                "error": error,
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        selectable = comparison[comparison["cv_metric_value"].replace([np.inf, -np.inf], np.nan).notna()]
        if selectable.empty:
            selected_index = comparison.index[0]
        else:
            selected_index = selectable["cv_metric_value"].astype(float).idxmin()
        comparison.loc[selected_index, "selected"] = True
        selected_model = str(comparison.loc[selected_index, "model_name"])
        selected_predictions = predictions[predictions["model_name"] == selected_model].copy()
        selected_predictions.index = pd.to_datetime(selected_predictions["date"])
        primary_signal = (
            selected_predictions["direction_signal"].groupby(level=0).last().reindex(df.index).fillna(0.0)
        )
    else:
        primary_signal = pd.Series(0.0, index=df.index)

    diagnostics = pd.DataFrame(
        {
            "date": df.index.date.astype(str),
            "hmm_signal": df.get("hmm_signal", pd.Series(0.0, index=df.index)).values,
            "forecast_signal": primary_signal.values,
            "garch_risk_multiplier": df.get("vol_risk_multiplier", pd.Series(1.0, index=df.index)).values,
            "final_confirmation_flag": True,
        },
        index=df.index,
    )
    return {
        "forecast_predictions": predictions,
        "forecast_model_comparison": comparison,
        "forecast_confirmation_diagnostics": diagnostics,
    }
