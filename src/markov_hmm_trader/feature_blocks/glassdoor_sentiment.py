from __future__ import annotations

import pandas as pd

from ..alt_data.glassdoor import build_glassdoor_features
from ..alt_data.loaders import load_glassdoor_reviews
from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata, skipped_feature_group


def build_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> pd.DataFrame:
    return build_glassdoor_sentiment(base, cfg, provider).features


def build_glassdoor_sentiment(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    reviews, warnings = load_glassdoor_reviews(cfg.glassdoor_reviews_path)
    if reviews.empty:
        skipped = skipped_feature_group("glassdoor_sentiment", "glassdoor_file_missing", "reviews")
        skipped.warnings = pd.concat([skipped.warnings, warnings], ignore_index=True)
        return skipped
    reviews = reviews[reviews["ticker"].astype(str).str.upper().eq(cfg.ticker.upper())]
    features = build_glassdoor_features(base, reviews, cfg)
    if features["n_reviews"].max(skipna=True) < cfg.glassdoor_min_reviews:
        insufficient = skipped_feature_group(
            "glassdoor_sentiment",
            "glassdoor_insufficient_reviews",
            f"min_reviews={cfg.glassdoor_min_reviews}",
        )
        warnings = pd.concat([warnings, insufficient.warnings], ignore_index=True)
    return FeatureBlockResult(
        name="glassdoor_sentiment",
        features=features,
        metadata=feature_metadata("glassdoor_sentiment", list(features.columns)),
        warnings=warnings,
        produced_columns=list(features.columns),
    )
