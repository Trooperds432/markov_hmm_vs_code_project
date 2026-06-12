from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata, skipped_feature_group
from .utils import asof_to_base, date_indexed


def _first_available(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name].astype(float)
    return pd.Series(np.nan, index=frame.index)


def derive_fundamental_features(
    fundamentals: pd.DataFrame | dict[str, pd.DataFrame], base: pd.DataFrame
) -> pd.DataFrame:
    if isinstance(fundamentals, dict):
        frames = []
        for _, frame in fundamentals.items():
            if not frame.empty:
                frames.append(date_indexed(frame))
        data = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    else:
        data = date_indexed(fundamentals)
    if data.empty:
        return pd.DataFrame(index=base.index)

    revenue = _first_available(data, ("revenue", "total_revenue", "Total Revenue"))
    net_income = _first_available(data, ("net_income", "Net Income"))
    debt = _first_available(data, ("debt", "total_debt", "Total Debt"))
    equity = _first_available(data, ("equity", "shareholder_equity", "Stockholders Equity"))
    fcf = _first_available(data, ("free_cash_flow", "free_cashflow", "Free Cash Flow"))
    eps_actual = _first_available(data, ("eps_actual", "actual_eps"))
    eps_estimate = _first_available(data, ("eps_estimate", "estimated_eps"))
    estimate = _first_available(data, ("estimate", "eps_estimate", "estimated_eps"))
    filing_count = _first_available(data, ("filing_count", "filings"))

    out = pd.DataFrame(index=data.index)
    out["earnings_surprise_proxy"] = (eps_actual - eps_estimate) / eps_estimate.abs().replace(
        0.0, np.nan
    )
    out["estimate_revision_momentum"] = estimate.pct_change(2)
    earnings_dates = out.index[out["earnings_surprise_proxy"].notna()]
    out["days_since_earnings"] = [
        min(
            [(date - event_date).days for event_date in earnings_dates if event_date <= date]
            or [np.nan]
        )
        for date in out.index
    ]
    out["filing_count_30d"] = filing_count.rolling(30, min_periods=1).sum()
    out["revenue_growth_proxy"] = revenue.pct_change(4)
    out["profit_margin_proxy"] = net_income / revenue.replace(0.0, np.nan)
    out["debt_to_equity_proxy"] = debt / equity.replace(0.0, np.nan)
    out["free_cashflow_margin_proxy"] = fcf / revenue.replace(0.0, np.nan)
    quality_inputs = pd.concat(
        [
            out["profit_margin_proxy"].rank(pct=True),
            out["free_cashflow_margin_proxy"].rank(pct=True),
            (-out["debt_to_equity_proxy"]).rank(pct=True),
        ],
        axis=1,
    )
    out["quality_score_proxy"] = quality_inputs.mean(axis=1)
    return asof_to_base(out.replace([np.inf, -np.inf], np.nan), base.index)


def build_fundamental_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    if provider is None or not provider.supports("fundamentals"):
        return skipped_feature_group("fundamentals", "provider_lacks_capability", "fundamentals")
    result = provider.get_fundamentals(cfg.ticker, cfg.start, cfg.end)
    if result.data.empty:
        skipped = skipped_feature_group("fundamentals", "empty_provider_result", "fundamentals")
        skipped.warnings = pd.concat([skipped.warnings, result.warnings], ignore_index=True)
        return skipped
    features = derive_fundamental_features(result.data, base)
    if features.empty:
        return skipped_feature_group("fundamentals", "empty_provider_result", "fundamentals")
    columns = list(features.columns)
    return FeatureBlockResult(
        name="fundamentals",
        features=features,
        metadata=feature_metadata("fundamentals", columns),
        warnings=result.warnings,
        produced_columns=columns,
    )
