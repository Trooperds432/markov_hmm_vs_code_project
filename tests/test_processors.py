from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.markov_hmm_trader.processors import (
    AsOfJoinProcessor,
    FeatureColumnFilter,
    LeakageGuardProcessor,
    ProcessorChain,
    RollingZScoreProcessor,
    WinsorizeProcessor,
    ZScoreProcessor,
)


def test_processor_chain_fit_transform_applies_steps_in_order():
    df = pd.DataFrame({"x": [1.0, 2.0, 100.0], "y": [5.0, 6.0, 7.0]})
    chain = ProcessorChain(
        [
            WinsorizeProcessor(columns=["x"], lower_quantile=0.0, upper_quantile=0.5),
            ZScoreProcessor(columns=["x"]),
            FeatureColumnFilter(columns=["x"]),
        ]
    )

    out = chain.fit_transform(df)

    assert list(out.columns) == ["x"]
    assert np.isclose(out["x"].mean(), 0.0)


def test_rolling_zscore_uses_prior_window():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 10.0]}, index=pd.RangeIndex(4))

    out = RollingZScoreProcessor(columns=["x"], window=3, shift=1).fit_transform(df)

    assert np.isnan(out.loc[2, "x_rz3"])
    assert out.loc[3, "x_rz3"] > 0.0


def test_asof_join_uses_backward_point_in_time_semantics():
    left = pd.DataFrame(index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]))
    right = pd.DataFrame(
        {"macro_value": [10.0, 20.0]},
        index=pd.to_datetime(["2020-01-02", "2020-01-05"]),
    )

    out = AsOfJoinProcessor(right).fit_transform(left)

    assert pd.isna(out.loc[pd.Timestamp("2020-01-01"), "macro_value"])
    assert out.loc[pd.Timestamp("2020-01-02"), "macro_value"] == 10.0
    assert out.loc[pd.Timestamp("2020-01-03"), "macro_value"] == 10.0


def test_leakage_guard_processor_rejects_future_like_columns():
    df = pd.DataFrame({"future_ret_5": [0.1, 0.2]})

    with pytest.raises(ValueError, match="future_like_feature_columns"):
        LeakageGuardProcessor(["future_ret_5"]).fit(df)
