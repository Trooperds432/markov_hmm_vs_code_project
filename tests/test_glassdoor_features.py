from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.markov_hmm_trader.alt_data.glassdoor import build_glassdoor_features
from src.markov_hmm_trader.alt_data.loaders import load_glassdoor_reviews
from src.markov_hmm_trader.config import PipelineConfig

FIXTURES = Path(__file__).parent / "fixtures"


def _market() -> pd.DataFrame:
    return pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=140))


def test_glassdoor_loads_aliases_and_builds_trailing_features():
    reviews, warnings = load_glassdoor_reviews(str(FIXTURES / "glassdoor_reviews_sample.csv"))
    cfg = PipelineConfig(ticker="SYNTH", glassdoor_publication_lag_days=7, glassdoor_min_reviews=2)

    features = build_glassdoor_features(_market(), reviews[reviews["ticker"].eq("SYNTH")], cfg)

    assert warnings.empty
    assert "glassdoor_sentiment_score" in features
    assert features["n_reviews"].max() >= 2
    assert features["days_since_last_review"].dropna().ge(0).all()
    assert features.loc[pd.Timestamp("2020-01-06"), "n_reviews"] == 0
    assert features.loc[pd.Timestamp("2020-01-13"), "n_reviews"] >= 1


def test_glassdoor_future_reviews_do_not_change_past_features():
    reviews, _ = load_glassdoor_reviews(str(FIXTURES / "glassdoor_reviews_sample.csv"))
    cfg = PipelineConfig(ticker="SYNTH", glassdoor_publication_lag_days=7)
    base = _market()
    original = build_glassdoor_features(base, reviews[reviews["ticker"].eq("SYNTH")], cfg)
    changed_reviews = reviews.copy()
    changed_reviews.loc[len(changed_reviews)] = [
        "SYNTH",
        pd.Timestamp("2021-01-01"),
        1,
        1,
        1,
        1,
        "negative",
        "negative",
        "negative",
    ]
    changed = build_glassdoor_features(
        base, changed_reviews[changed_reviews["ticker"].eq("SYNTH")], cfg
    )

    pd.testing.assert_series_equal(
        original["glassdoor_stress_score"].iloc[:100],
        changed["glassdoor_stress_score"].iloc[:100],
    )
