from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.feature_quality import apply_feature_quality


def test_feature_quality_drops_bad_features_and_winsorizes():
    index = pd.bdate_range("2020-01-01", periods=80)
    frame = pd.DataFrame(
        {
            "good": np.r_[np.arange(79, dtype=float), 10_000.0],
            "high_missing": [np.nan] * 70 + list(range(10)),
            "zero_var": 1.0,
            "future_return_label": np.arange(80, dtype=float),
        },
        index=index,
    )
    cfg = PipelineConfig(
        max_feature_missing_ratio=0.30,
        feature_winsorize_limits=(0.05, 0.95),
    )

    result = apply_feature_quality(
        frame,
        cfg,
        ["good", "high_missing", "zero_var", "future_return_label"],
    )

    assert result.hmm_features == ["good"]
    drops = result.drop_report.set_index("feature")["drop_reason"].to_dict()
    assert drops["high_missing"] == "high_missing_ratio"
    assert drops["zero_var"] == "zero_variance"
    assert drops["future_return_label"] == "validation_only"
    assert result.frame["good"].max() < 10_000.0
