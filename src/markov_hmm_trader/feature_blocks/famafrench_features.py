from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata, skipped_feature_group
from .utils import asof_to_base, date_indexed


def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    covariance = y.rolling(window).cov(x).shift(1)
    variance = x.rolling(window).var(ddof=0).shift(1)
    return covariance / variance.replace(0.0, np.nan)


def derive_famafrench_features(factors: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    data = date_indexed(factors)
    if data.empty:
        return pd.DataFrame(index=base.index)
    rename = {
        "Mkt-RF": "ff_market_factor",
        "mkt_rf": "ff_market_factor",
        "market": "ff_market_factor",
        "SMB": "ff_smb",
        "smb": "ff_smb",
        "HML": "ff_hml",
        "hml": "ff_hml",
        "MOM": "ff_mom",
        "mom": "ff_mom",
        "UMD": "ff_mom",
    }
    data = data.rename(columns={key: value for key, value in rename.items() if key in data})
    out = pd.DataFrame(index=data.index)
    for column in ("ff_market_factor", "ff_smb", "ff_hml", "ff_mom"):
        out[column] = data.get(column, pd.Series(np.nan, index=data.index)).astype(float)
    aligned = asof_to_base(out, base.index)
    asset_ret = base["Close"].astype(float).pct_change()
    aligned["rolling_beta_ff_market_60"] = _rolling_beta(asset_ret, aligned["ff_market_factor"], 60)
    aligned["rolling_beta_ff_smb_60"] = _rolling_beta(asset_ret, aligned["ff_smb"], 60)
    aligned["rolling_beta_ff_hml_60"] = _rolling_beta(asset_ret, aligned["ff_hml"], 60)
    factor_return = (
        aligned["rolling_beta_ff_market_60"] * aligned["ff_market_factor"]
        + aligned["rolling_beta_ff_smb_60"] * aligned["ff_smb"]
        + aligned["rolling_beta_ff_hml_60"] * aligned["ff_hml"]
    )
    aligned["factor_residual_return"] = asset_ret - factor_return
    return aligned.replace([np.inf, -np.inf], np.nan)


def build_famafrench_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    if provider is None or not provider.supports("famafrench"):
        return skipped_feature_group("famafrench", "provider_lacks_capability", "famafrench")
    result = provider.get_fama_french("factors", cfg.start, cfg.end)
    if result.data.empty:
        skipped = skipped_feature_group("famafrench", "empty_provider_result", "famafrench")
        skipped.warnings = pd.concat([skipped.warnings, result.warnings], ignore_index=True)
        return skipped
    features = derive_famafrench_features(result.data, base)
    columns = list(features.columns)
    return FeatureBlockResult(
        name="famafrench",
        features=features,
        metadata=feature_metadata("famafrench", columns),
        warnings=result.warnings,
        produced_columns=columns,
    )
