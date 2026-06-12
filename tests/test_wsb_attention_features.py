from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.markov_hmm_trader.alt_data.loaders import load_wsb_mentions
from src.markov_hmm_trader.alt_data.wsb import build_wsb_attention_features
from src.markov_hmm_trader.config import PipelineConfig

FIXTURES = Path(__file__).parent / "fixtures"


def test_wsb_features_compute_mentions_and_attention_risk():
    mentions, warnings = load_wsb_mentions(str(FIXTURES / "wsb_mentions_sample.csv"))
    market = pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=80))
    cfg = PipelineConfig(ticker="SYNTH", wsb_publication_lag_days=1)

    features = build_wsb_attention_features(market, mentions, cfg)

    assert warnings.empty
    assert {"wsb_mentions", "wsb_mentions_z_20d", "wsb_meme_risk_score"}.issubset(features)
    assert features["wsb_mentions"].max() >= 20
    assert features["wsb_rank_percentile"].dropna().between(0.0, 1.0).all()


def test_wsb_future_mentions_do_not_change_past_features():
    mentions, _ = load_wsb_mentions(str(FIXTURES / "wsb_mentions_sample.csv"))
    market = pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=80))
    cfg = PipelineConfig(ticker="SYNTH", wsb_publication_lag_days=1)
    original = build_wsb_attention_features(market, mentions, cfg)
    future = pd.concat(
        [
            mentions,
            pd.DataFrame(
                [{"date": pd.Timestamp("2021-01-01"), "ticker": "SYNTH", "mention_count": 9999}]
            ),
        ],
        ignore_index=True,
    )
    changed = build_wsb_attention_features(market, future, cfg)

    pd.testing.assert_series_equal(
        original["wsb_mentions"].iloc[:60], changed["wsb_mentions"].iloc[:60]
    )
