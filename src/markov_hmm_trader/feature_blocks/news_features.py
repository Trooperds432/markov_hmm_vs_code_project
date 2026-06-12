from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata, skipped_feature_group
from .utils import asof_to_base, date_indexed

POSITIVE_WORDS = {
    "beat",
    "beats",
    "growth",
    "upgrade",
    "record",
    "strong",
    "profit",
    "surge",
    "optimistic",
}
NEGATIVE_WORDS = {
    "miss",
    "misses",
    "downgrade",
    "lawsuit",
    "probe",
    "weak",
    "loss",
    "plunge",
    "warning",
}
UNCERTAINTY_WORDS = {"uncertain", "risk", "volatility", "delay", "concern", "guidance"}
REGULATORY_WORDS = {"sec", "regulator", "regulatory", "antitrust", "lawsuit", "probe"}
EARNINGS_WORDS = {"earnings", "revenue", "margin", "guidance", "eps"}
WORLD_RISK_WORDS = {"war", "sanction", "tariff", "crisis", "recession"}


def _sentiment(text: str) -> float:
    words = set(re.findall(r"[a-z]+", str(text).lower()))
    return float(len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS))


def _ratio(text: str, lexicon: set[str]) -> float:
    words = re.findall(r"[a-z]+", str(text).lower())
    if not words:
        return 0.0
    return float(sum(word in lexicon for word in words) / len(words))


def derive_news_features(news: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(index=base.index)
    data = date_indexed(news, ("published_at", "date", "Date", "created_at"))
    headline_col = "headline" if "headline" in data else "title" if "title" in data else None
    if headline_col is None:
        return pd.DataFrame(index=base.index)
    data = data.copy()
    data["_sentiment"] = data[headline_col].map(_sentiment)
    data["_bad_news"] = (data["_sentiment"] < 0).astype(float)
    data["_uncertainty"] = data[headline_col].map(lambda value: _ratio(value, UNCERTAINTY_WORDS))
    data["_regulatory"] = data[headline_col].map(
        lambda value: float(_ratio(value, REGULATORY_WORDS) > 0)
    )
    data["_earnings"] = data[headline_col].map(
        lambda value: float(_ratio(value, EARNINGS_WORDS) > 0)
    )
    data["_world_risk"] = data[headline_col].map(
        lambda value: float(_ratio(value, WORLD_RISK_WORDS) > 0)
    )
    daily = data.groupby(data.index).agg(
        news_count=("headline" if headline_col == "headline" else headline_col, "count"),
        headline_sentiment_mean=("_sentiment", "mean"),
        headline_sentiment_std=("_sentiment", "std"),
        bad_news_ratio=("_bad_news", "mean"),
        uncertainty_keyword_ratio=("_uncertainty", "mean"),
        litigation_regulatory_keyword_flag=("_regulatory", "max"),
        earnings_keyword_flag=("_earnings", "max"),
        world_risk_news_count=("_world_risk", "sum"),
    )
    out = daily.copy()
    out["news_count_1d"] = out["news_count"]
    out["news_count_7d"] = out["news_count"].rolling(7, min_periods=1).sum()
    out["news_count_30d"] = out["news_count"].rolling(30, min_periods=1).sum()
    out = out.drop(columns=["news_count"])
    return asof_to_base(out.replace([np.inf, -np.inf], np.nan), base.index).fillna(0.0)


def build_news_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    if provider is None or not provider.supports("news"):
        return skipped_feature_group("news", "provider_lacks_capability", "news")
    result = provider.get_company_news(cfg.ticker, cfg.start, cfg.end)
    if result.data.empty:
        skipped = skipped_feature_group("news", "empty_provider_result", "news")
        skipped.warnings = pd.concat([skipped.warnings, result.warnings], ignore_index=True)
        return skipped
    features = derive_news_features(result.data, base)
    if features.empty:
        return skipped_feature_group("news", "missing_required_columns", "headline")
    columns = list(features.columns)
    return FeatureBlockResult(
        name="news",
        features=features,
        metadata=feature_metadata("news", columns),
        warnings=result.warnings,
        produced_columns=columns,
    )
