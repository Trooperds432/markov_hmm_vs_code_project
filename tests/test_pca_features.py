from __future__ import annotations

import numpy as np

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.feature_blocks.research_public_features import (
    build_research_public_features,
)
from tests.test_research_public_features import _research_frame


def test_pca_market_mode_features_are_bounded():
    cfg = PipelineConfig(
        enable_research_public_features=True,
        research_public_feature_set=("pca_market_mode",),
        pca_window=80,
    )

    result = build_research_public_features(_research_frame(220), cfg)
    explained = result.features["pc1_explained_variance"].dropna()

    assert not explained.empty
    assert explained.between(0.0, 1.0).all()
    assert set(result.features["pca_market_mode_flag"].dropna().unique()).issubset({0.0, 1.0})
    assert np.isfinite(result.features["pca_reconstruction_error"].dropna()).all()
