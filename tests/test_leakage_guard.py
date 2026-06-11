from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.leakage import FeatureLeakageGuard


def test_feature_target_alignment_accepts_past_features_future_targets():
    guard = FeatureLeakageGuard()
    decisions = pd.date_range("2020-01-01", periods=3)
    features = decisions
    targets = decisions + pd.Timedelta(days=1)

    result = guard.check_feature_target_alignment(features, decisions, targets)

    assert result.status == "pass"


def test_future_return_feature_is_detected():
    result = FeatureLeakageGuard().check_no_future_columns(["z_ret_1", "future_ret_20"])

    assert result.status == "fail"
    assert result.details["columns"] == ["future_ret_20"]


def test_scaler_fitted_on_full_data_is_detected():
    result = FeatureLeakageGuard().check_scaler_fit_scope(
        fit_indices=[1, 2, 3, 4],
        train_indices=[1, 2, 3],
    )

    assert result.status == "fail"


def test_target_not_shifted_forward_is_detected():
    guard = FeatureLeakageGuard()
    decisions = pd.date_range("2020-01-01", periods=3)

    result = guard.check_feature_target_alignment(decisions, decisions, decisions)

    assert result.status == "fail"


def test_insufficient_embargo_is_detected():
    result = FeatureLeakageGuard().check_purge_embargo(
        train_end_position=10,
        test_start_position=11,
        purge_bars=1,
        embargo_bars=1,
    )

    assert result.status == "fail"
