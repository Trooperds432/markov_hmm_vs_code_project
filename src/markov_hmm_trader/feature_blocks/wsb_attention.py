from __future__ import annotations

import pandas as pd

from ..alt_data.loaders import load_wsb_mentions
from ..alt_data.wsb import build_wsb_attention_features
from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata, skipped_feature_group


def build_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> pd.DataFrame:
    return build_wsb_attention(base, cfg, provider).features


def build_wsb_attention(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    mentions, warnings = load_wsb_mentions(cfg.wsb_mentions_path)
    if mentions.empty:
        skipped = skipped_feature_group("wsb_attention", "wsb_file_missing", "mentions")
        skipped.warnings = pd.concat([skipped.warnings, warnings], ignore_index=True)
        return skipped
    features = build_wsb_attention_features(base, mentions, cfg)
    if features["wsb_mentions"].sum(skipna=True) < cfg.wsb_min_mentions:
        insufficient = skipped_feature_group("wsb_attention", "wsb_no_mentions", "no_mentions")
        warnings = pd.concat([warnings, insufficient.warnings], ignore_index=True)
    return FeatureBlockResult(
        name="wsb_attention",
        features=features,
        metadata=feature_metadata("wsb_attention", list(features.columns)),
        warnings=warnings,
        produced_columns=list(features.columns),
    )
