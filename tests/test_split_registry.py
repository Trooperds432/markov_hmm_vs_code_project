from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.splits import SPLIT_REGISTRY_COLUMNS, build_split_registry


def test_split_registry_is_deterministic_and_contract_compliant():
    index = pd.bdate_range("2020-01-01", periods=140)

    first = build_split_registry(
        index,
        symbol="SYNTH",
        train_window=60,
        test_window=20,
        validation_fraction=0.2,
        purge_bars=1,
        embargo_bars=2,
        created_at_utc="2020-01-01T00:00:00Z",
    )
    second = build_split_registry(
        index,
        symbol="SYNTH",
        train_window=60,
        test_window=20,
        validation_fraction=0.2,
        purge_bars=1,
        embargo_bars=2,
        created_at_utc="2020-01-01T00:00:00Z",
    )

    assert list(first.columns) == SPLIT_REGISTRY_COLUMNS
    assert not first.empty
    assert first.equals(second)
    assert first.loc[0, "purge_bars"] == 1
    assert first.loc[0, "embargo_bars"] == 2
