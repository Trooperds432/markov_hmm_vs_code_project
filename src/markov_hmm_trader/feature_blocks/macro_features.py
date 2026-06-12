from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata, skipped_feature_group
from .utils import asof_to_base, date_indexed, safe_log_return


def derive_macro_features(macro: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    data = date_indexed(macro)
    if {"series_id", "value"}.issubset(data.columns):
        data = data.pivot_table(
            index=data.index, columns="series_id", values="value", aggfunc="last"
        )
    out = pd.DataFrame(index=data.index)
    ten_year = data.get("10y", data.get("DGS10", pd.Series(np.nan, index=data.index)))
    two_year = data.get("2y", data.get("DGS2", pd.Series(np.nan, index=data.index)))
    short_rate = data.get("short_rate", data.get("DGS3MO", pd.Series(np.nan, index=data.index)))
    inflation = data.get("inflation", data.get("CPI", pd.Series(np.nan, index=data.index)))
    unemployment = data.get("unemployment", data.get("UNRATE", pd.Series(np.nan, index=data.index)))
    growth = data.get("growth", data.get("GDP", pd.Series(np.nan, index=data.index)))
    credit = data.get("credit_spread", pd.Series(np.nan, index=data.index))
    dollar = data.get("dollar", pd.Series(np.nan, index=data.index))
    rates = data.get("rates", pd.Series(np.nan, index=data.index))
    event_flag = data.get("event_flag", pd.Series(0.0, index=data.index)).fillna(0.0)

    out["yield_curve_10y_2y"] = ten_year.astype(float) - two_year.astype(float)
    out["short_rate_change_20"] = short_rate.astype(float).diff(20)
    out["inflation_proxy_momentum"] = inflation.astype(float).pct_change(20)
    out["unemployment_proxy_change"] = unemployment.astype(float).diff(20)
    out["growth_proxy_momentum"] = growth.astype(float).pct_change(20)
    out["credit_spread_proxy"] = credit.astype(float)
    out["dollar_momentum_20"] = dollar.astype(float).pct_change(20)
    out["rates_momentum_20"] = rates.astype(float).pct_change(20)
    out["economic_calendar_event_flag"] = event_flag.astype(float).clip(0.0, 1.0)
    event_dates = out.index[out["economic_calendar_event_flag"] > 0]
    out["days_since_major_macro_release"] = [
        min(
            [(date - event_date).days for event_date in event_dates if event_date <= date]
            or [np.nan]
        )
        for date in out.index
    ]
    risk_inputs = out[
        [
            "yield_curve_10y_2y",
            "short_rate_change_20",
            "inflation_proxy_momentum",
            "credit_spread_proxy",
            "dollar_momentum_20",
            "rates_momentum_20",
        ]
    ]
    out["macro_risk_score"] = risk_inputs.rank(pct=True).mean(axis=1)
    return asof_to_base(out.replace([np.inf, -np.inf], np.nan), base.index)


def _proxy_macro_features(base: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=base.index)
    out["dollar_momentum_20"] = safe_log_return(base, "dollar_proxy_Close", 20)
    out["rates_momentum_20"] = safe_log_return(base, "rates_proxy_Close", 20)
    credit = safe_log_return(base, "credit_proxy_Close", 20)
    defensive = safe_log_return(base, "defensive_proxy_Close", 20)
    growth = safe_log_return(base, "growth_proxy_Close", 20)
    out["credit_spread_proxy"] = -credit
    out["yield_curve_10y_2y"] = -out["rates_momentum_20"]
    out["short_rate_change_20"] = out["rates_momentum_20"]
    out["inflation_proxy_momentum"] = pd.Series(np.nan, index=base.index)
    out["unemployment_proxy_change"] = pd.Series(np.nan, index=base.index)
    out["growth_proxy_momentum"] = growth
    out["economic_calendar_event_flag"] = 0.0
    out["days_since_major_macro_release"] = np.nan
    out["macro_risk_score"] = (
        pd.concat(
            [-credit, out["rates_momentum_20"], out["dollar_momentum_20"], defensive - growth],
            axis=1,
        )
        .rank(pct=True)
        .mean(axis=1)
    )
    return out.replace([np.inf, -np.inf], np.nan)


def build_macro_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    proxy_columns = {
        "dollar_proxy_Close",
        "rates_proxy_Close",
        "credit_proxy_Close",
        "defensive_proxy_Close",
        "growth_proxy_Close",
    }
    if proxy_columns.intersection(base.columns):
        features = _proxy_macro_features(base)
        columns = list(features.columns)
        return FeatureBlockResult(
            name="macro",
            features=features,
            metadata=feature_metadata("macro", columns),
            produced_columns=columns,
        )
    if provider is None or not provider.supports("macro"):
        return skipped_feature_group("macro", "provider_lacks_capability", "macro")
    result = provider.get_macro_series("macro", cfg.start, cfg.end)
    if result.data.empty:
        skipped = skipped_feature_group("macro", "empty_provider_result", "macro")
        skipped.warnings = pd.concat([skipped.warnings, result.warnings], ignore_index=True)
        return skipped
    features = derive_macro_features(result.data, base)
    columns = list(features.columns)
    return FeatureBlockResult(
        name="macro",
        features=features,
        metadata=feature_metadata("macro", columns),
        warnings=result.warnings,
        produced_columns=columns,
    )
