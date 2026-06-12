from __future__ import annotations

import pandas as pd

from ..alt_data.overlays import compute_alt_data_overlay
from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata


def build_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> pd.DataFrame:
    return build_alt_data_overlays(base, cfg, provider).features


def build_alt_data_overlays(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    features = pd.DataFrame(
        [compute_alt_data_overlay(row, cfg) for _, row in base.iterrows()],
        index=base.index,
    )
    return FeatureBlockResult(
        name="alt_data_overlays",
        features=features,
        metadata=feature_metadata("alt_data_overlays", list(features.columns)),
        produced_columns=list(features.columns),
    )
