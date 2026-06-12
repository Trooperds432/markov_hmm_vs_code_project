from __future__ import annotations

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.factor_validation import (
    factor_rank_diagnostics,
    long_short_spread_diagnostics,
)
from tests.test_research_public_features import _research_frame


def test_factor_validation_uses_forward_returns_only_in_diagnostics():
    frame = _research_frame()
    frame["rank_percentile"] = frame["Close"].pct_change(30).rank(pct=True)
    cfg = PipelineConfig(enable_research_public_features=True)

    diagnostics = factor_rank_diagnostics(frame, cfg)
    spread = long_short_spread_diagnostics(frame, cfg)

    assert {
        "top_decile_forward_return",
        "bottom_decile_forward_return",
        "long_short_spread_return",
        "factor_hit_rate",
    }.issubset(diagnostics.columns)
    assert diagnostics.iloc[0]["notes"] == "ex_post_forward_return_validation_not_hmm_input"
    assert "spread_name" in spread.columns
