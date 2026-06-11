from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, feature_metadata, skipped_feature_group


def build_breadth_features(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None,
) -> FeatureBlockResult:
    if provider is None or not provider.supports("breadth"):
        return skipped_feature_group("breadth", "provider_lacks_capability", "breadth")
    snapshots = provider.get_market_snapshots(list(cfg.universe[: min(len(cfg.universe), 20)]))
    if snapshots.data.empty:
        return skipped_feature_group("breadth", "empty_provider_result", "market_snapshots")
    data = snapshots.data.copy()
    if "return_20d" not in data or "vol_20d" not in data:
        return skipped_feature_group("breadth", "missing_snapshot_columns", "return_20d,vol_20d")
    values = {
        "breadth_percent_positive": float((data["return_20d"] > 0).mean()),
        "breadth_return_dispersion": float(data["return_20d"].std(ddof=0)),
        "breadth_peer_relative_return_rank": float(data["return_20d"].rank(pct=True).mean()),
        "breadth_peer_relative_vol_rank": float(data["vol_20d"].rank(pct=True).mean()),
        "breadth_top_bottom_concentration": float(
            data["return_20d"].nlargest(3).sum() - data["return_20d"].nsmallest(3).sum()
        ),
    }
    out = pd.DataFrame(
        {column: np.repeat(value, len(df)) for column, value in values.items()}, index=df.index
    )
    columns = list(out.columns)
    result = FeatureBlockResult(
        name="breadth",
        features=out,
        metadata=feature_metadata("breadth", columns),
        warnings=snapshots.warnings,
        produced_columns=columns,
    )
    return result
