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
    groups = list(
        dict.fromkeys(
            canonical_feature_group(group)
            for group in (*cfg.feature_groups_enabled, *cfg.optional_feature_groups)
        )
    )
    enabled = {canonical_feature_group(item) for item in cfg.feature_groups_enabled}
    feature_frames = []
    metadata_frames = []
    warning_frames = []
    produced_columns: list[str] = []
    skipped: list[dict[str, str]] = []
    for group in groups:
        if group in BUILDERS:
            if group not in enabled:
                continue
            result = BUILDERS[group](df, cfg, provider)
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
