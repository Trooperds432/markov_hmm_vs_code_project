from __future__ import annotations

import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult
from .breadth_features import build_breadth_features
from .optional_blocks import CAPABILITY_BY_GROUP, build_optional_skip_group
from .price_core import build_price_core
from .qlib_alpha_features import build_alpha158_lite
from .technical_features import build_technical_features
from .volatility_core import build_volatility_core

BUILDERS = {
    "price_core": lambda df, cfg, provider: build_price_core(df),
    "volatility_core": lambda df, cfg, provider: build_volatility_core(df),
    "technical": lambda df, cfg, provider: build_technical_features(df),
    "alpha158_lite": lambda df, cfg, provider: build_alpha158_lite(df),
    "breadth": lambda df, cfg, provider: build_breadth_features(df, cfg, provider),
}


def build_configured_feature_groups(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    groups = list(dict.fromkeys((*cfg.feature_groups_enabled, *cfg.optional_feature_groups)))
    feature_frames = []
    metadata_frames = []
    warning_frames = []
    produced_columns: list[str] = []
    skipped: list[dict[str, str]] = []
    for group in groups:
        if group in BUILDERS:
            if group not in cfg.feature_groups_enabled:
                continue
            result = BUILDERS[group](df, cfg, provider)
        elif group in CAPABILITY_BY_GROUP and group in cfg.feature_groups_enabled:
            result = build_optional_skip_group(group, provider)
        else:
            continue
        if not result.features.empty:
            feature_frames.append(result.features)
        if not result.metadata.empty:
            metadata_frames.append(result.metadata)
        if not result.warnings.empty:
            warning_frames.append(result.warnings)
        produced_columns.extend(result.produced_columns)
        skipped.extend(result.skipped)

    features = pd.concat(feature_frames, axis=1) if feature_frames else pd.DataFrame(index=df.index)
    metadata = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    warnings = pd.concat(warning_frames, ignore_index=True) if warning_frames else pd.DataFrame()
    return FeatureBlockResult(
        name="configured_feature_groups",
        features=features,
        metadata=metadata,
        warnings=warnings,
        produced_columns=produced_columns,
        skipped=skipped,
    )
