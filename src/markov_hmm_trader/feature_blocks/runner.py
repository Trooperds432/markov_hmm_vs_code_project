from __future__ import annotations

import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult
from .registry import BUILDERS, canonical_feature_group


def build_configured_feature_groups(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    configured_groups = list(cfg.feature_groups_enabled)
    if (
        cfg.enable_research_public_features or cfg.use_research_public_hmm_features
    ) and "research_public" not in configured_groups:
        configured_groups.append("research_public")
    alt_auto_groups = []
    if cfg.enable_alt_data_features or cfg.enable_glassdoor_features:
        alt_auto_groups.append("glassdoor_sentiment")
    if cfg.enable_alt_data_features or cfg.enable_wsb_features:
        alt_auto_groups.append("wsb_attention")
    if cfg.enable_alt_data_features or cfg.enable_valuation_features:
        alt_auto_groups.append("valuation_anchor")
    if (
        cfg.enable_alt_data_features
        or cfg.use_alt_data_position_overlay
        or cfg.use_alt_data_portfolio_ranking
    ):
        alt_auto_groups.append("alt_data_overlays")
    for group in alt_auto_groups:
        if group not in configured_groups:
            configured_groups.append(group)
    groups = list(
        dict.fromkeys(
            canonical_feature_group(group)
            for group in (*configured_groups, *cfg.optional_feature_groups)
        )
    )
    enabled = {canonical_feature_group(item) for item in configured_groups}
    feature_frames = []
    metadata_frames = []
    warning_frames = []
    produced_columns: list[str] = []
    skipped: list[dict[str, str]] = []
    current = df.copy()
    for group in groups:
        if group in BUILDERS:
            if group not in enabled:
                continue
            result = BUILDERS[group](current, cfg, provider)
        else:
            continue
        if not result.features.empty:
            feature_frames.append(result.features)
            current = pd.concat([current, result.features], axis=1)
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
