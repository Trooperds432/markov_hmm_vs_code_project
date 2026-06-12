from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.alt_data.joins import safe_asof_join


def test_safe_asof_join_applies_publication_lag_and_uses_past_only():
    market = pd.DataFrame(index=pd.bdate_range("2020-01-01", periods=6))
    alt = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-06"]),
            "ticker": ["ABC", "ABC"],
            "value": [1.0, 99.0],
        }
    )

    joined = safe_asof_join(market, alt, "ABC", "date", lag_days=2)

    assert pd.isna(joined.loc[pd.Timestamp("2020-01-03"), "value"])
    assert joined.loc[pd.Timestamp("2020-01-06"), "value"] == 1.0
    assert joined.loc[pd.Timestamp("2020-01-08"), "value"] == 99.0
