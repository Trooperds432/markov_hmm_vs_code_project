from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata, skipped_feature_group
from .utils import asof_to_base, date_indexed, safe_zscore


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    lower = {column.lower(): column for column in frame.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def derive_options_features(chain: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame(index=base.index)
    data = date_indexed(chain)
    option_type_col = _column(data, "option_type", "type", "put_call")
    strike_col = _column(data, "strike")
    iv_col = _column(data, "implied_volatility", "iv", "impliedVolatility")
    volume_col = _column(data, "volume")
    oi_col = _column(data, "open_interest", "openInterest", "oi")
    delta_col = _column(data, "delta")
    gamma_col = _column(data, "gamma")
    vega_col = _column(data, "vega")
    underlying_col = _column(data, "underlying_price", "underlying", "spot", "Close")
    expiry_col = _column(data, "expiration", "expiry", "expiration_date")

    if iv_col is None:
        return pd.DataFrame(index=base.index)

    rows = []
    for date, group in data.groupby(level=0):
        local = group.copy()
        option_type = (
            local[option_type_col].astype(str).str.lower()
            if option_type_col
            else pd.Series("", index=local.index)
        )
        calls = local[option_type.str.startswith("c")]
        puts = local[option_type.str.startswith("p")]
        spot = (
            float(local[underlying_col].dropna().iloc[0])
            if underlying_col and local[underlying_col].notna().any()
            else np.nan
        )
        if not np.isfinite(spot):
            spot = float(base["Close"].reindex([date], method="ffill").iloc[0])
        atm = local
        if strike_col and np.isfinite(spot):
            atm = local.loc[(local[strike_col].astype(float) - spot).abs().sort_values().index[:4]]

        near_iv = float(atm[iv_col].astype(float).mean())
        far_iv = near_iv
        if expiry_col:
            expiries = pd.to_datetime(local[expiry_col], errors="coerce")
            local = local.assign(_days=(expiries - pd.Timestamp(date)).dt.days)
            near = local[local["_days"] == local["_days"].dropna().min()]
            far = local[local["_days"] == local["_days"].dropna().max()]
            if not near.empty:
                near_iv = float(near[iv_col].astype(float).mean())
            if not far.empty:
                far_iv = float(far[iv_col].astype(float).mean())

        call_oi = calls[oi_col].astype(float).sum() if oi_col and not calls.empty else np.nan
        put_oi = puts[oi_col].astype(float).sum() if oi_col and not puts.empty else np.nan
        call_vol = (
            calls[volume_col].astype(float).sum() if volume_col and not calls.empty else np.nan
        )
        put_vol = puts[volume_col].astype(float).sum() if volume_col and not puts.empty else np.nan

        skew = np.nan
        if delta_col and not calls.empty and not puts.empty:
            call_25 = calls.iloc[(calls[delta_col].abs().astype(float) - 0.25).abs().argsort()[:1]]
            put_25 = puts.iloc[(puts[delta_col].abs().astype(float) - 0.25).abs().argsort()[:1]]
            if not call_25.empty and not put_25.empty:
                skew = float(
                    put_25[iv_col].astype(float).iloc[0] - call_25[iv_col].astype(float).iloc[0]
                )

        rows.append(
            {
                "date": pd.Timestamp(date),
                "atm_iv_near": near_iv,
                "atm_iv_far": far_iv,
                "iv_term_slope": far_iv - near_iv,
                "put_call_oi_ratio": (
                    put_oi / call_oi if call_oi and np.isfinite(call_oi) else np.nan
                ),
                "put_call_volume_ratio": (
                    put_vol / call_vol if call_vol and np.isfinite(call_vol) else np.nan
                ),
                "call_iv_mean": (
                    float(calls[iv_col].astype(float).mean()) if not calls.empty else np.nan
                ),
                "put_iv_mean": (
                    float(puts[iv_col].astype(float).mean()) if not puts.empty else np.nan
                ),
                "iv_skew_25d_proxy": skew,
                "gamma_exposure_proxy": (
                    float((local.get(gamma_col, 0.0) * local.get(oi_col, 0.0)).sum())
                    if gamma_col and oi_col
                    else np.nan
                ),
                "vega_exposure_proxy": (
                    float((local.get(vega_col, 0.0) * local.get(oi_col, 0.0)).sum())
                    if vega_col and oi_col
                    else np.nan
                ),
                "options_volume": (
                    float(local[volume_col].astype(float).sum()) if volume_col else np.nan
                ),
                "options_oi": float(local[oi_col].astype(float).sum()) if oi_col else np.nan,
            }
        )
    daily = pd.DataFrame(rows).set_index("date").sort_index()
    daily["options_volume_zscore"] = safe_zscore(daily["options_volume"], 20)
    daily["options_oi_zscore"] = safe_zscore(daily["options_oi"], 20)
    daily = daily.drop(columns=["options_volume", "options_oi"])
    return asof_to_base(daily, base.index).replace([np.inf, -np.inf], np.nan)


def build_options_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    if provider is None or not provider.supports("options"):
        return skipped_feature_group("options", "provider_lacks_capability", "options")
    result = provider.get_options_chain(cfg.ticker)
    if result.data.empty:
        skipped = skipped_feature_group("options", "empty_provider_result", "options_chain")
        skipped.warnings = pd.concat([skipped.warnings, result.warnings], ignore_index=True)
        return skipped
    features = derive_options_features(result.data, base)
    if features.empty:
        return skipped_feature_group("options", "missing_required_columns", "implied_volatility")
    columns = list(features.columns)
    return FeatureBlockResult(
        name="options",
        features=features,
        metadata=feature_metadata("options", columns),
        warnings=result.warnings,
        produced_columns=columns,
    )
