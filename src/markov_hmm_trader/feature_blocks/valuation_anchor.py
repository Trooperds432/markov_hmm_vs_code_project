from __future__ import annotations

import pandas as pd

from ..alt_data.loaders import load_valuation_data
from ..alt_data.valuation import build_valuation_features
from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata, skipped_feature_group


def build_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> pd.DataFrame:
    return build_valuation_anchor(base, cfg, provider).features


def build_valuation_anchor(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    valuation, warnings = load_valuation_data(cfg.valuation_data_path)
    if valuation.empty:
        skipped = skipped_feature_group("valuation_anchor", "valuation_file_missing", "valuation")
        skipped.warnings = pd.concat([skipped.warnings, warnings], ignore_index=True)
        return skipped
    features = build_valuation_features(base, valuation, cfg)
    if features["graham_intrinsic_value"].isna().all():
        invalid = skipped_feature_group(
            "valuation_anchor", "valuation_invalid_intrinsic_value", "intrinsic_value"
        )
        warnings = pd.concat([warnings, invalid.warnings], ignore_index=True)
    return FeatureBlockResult(
        name="valuation_anchor",
        features=features,
        metadata=feature_metadata("valuation_anchor", list(features.columns)),
        warnings=warnings,
        produced_columns=list(features.columns),
    )
