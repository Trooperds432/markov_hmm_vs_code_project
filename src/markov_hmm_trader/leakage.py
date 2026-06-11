from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeakageCheckResult:
    name: str
    status: str
    message: str
    severity: str = "error"
    details: dict[str, object] = field(default_factory=dict)


def _result(
    name: str,
    passed: bool,
    message: str,
    severity: str = "error",
    details: dict[str, object] | None = None,
) -> LeakageCheckResult:
    return LeakageCheckResult(
        name=name,
        status="pass" if passed else "fail",
        message=message,
        severity=severity,
        details=details or {},
    )


class FeatureLeakageGuard:
    future_column_tokens = ("future", "forward", "fwd", "target", "lead")

    def check_feature_target_alignment(
        self,
        feature_timestamps: Iterable[pd.Timestamp],
        decision_timestamps: Iterable[pd.Timestamp],
        target_timestamps: Iterable[pd.Timestamp],
    ) -> LeakageCheckResult:
        features = pd.to_datetime(pd.Index(feature_timestamps))
        decisions = pd.to_datetime(pd.Index(decision_timestamps))
        targets = pd.to_datetime(pd.Index(target_timestamps))
        n = min(len(features), len(decisions), len(targets))
        if n == 0:
            return _result("feature_target_alignment", False, "empty_alignment")

        features = features[:n]
        decisions = decisions[:n]
        targets = targets[:n]
        feature_ok = bool((features <= decisions).all())
        target_ok = bool((targets > decisions).all())
        return _result(
            "feature_target_alignment",
            feature_ok and target_ok,
            "ok" if feature_ok and target_ok else "feature_or_target_timestamp_leak",
            details={"rows_checked": n},
        )

    def check_no_future_columns(self, columns: Iterable[str]) -> LeakageCheckResult:
        leaked = [
            column
            for column in columns
            if any(token in column.lower() for token in self.future_column_tokens)
        ]
        return _result(
            "no_future_columns",
            not leaked,
            "ok" if not leaked else "future_like_feature_columns",
            details={"columns": leaked},
        )

    def check_train_validation_test_boundaries(
        self,
        train_end: pd.Timestamp,
        validation_start: pd.Timestamp,
        validation_end: pd.Timestamp,
        test_start: pd.Timestamp,
    ) -> LeakageCheckResult:
        passed = train_end < validation_start <= validation_end < test_start
        return _result(
            "train_validation_test_boundaries",
            bool(passed),
            "ok" if passed else "overlapping_or_unordered_splits",
        )

    def check_rolling_window_shift(
        self,
        feature: pd.Series,
        raw: pd.Series,
        window: int,
    ) -> LeakageCheckResult:
        aligned = pd.concat([feature, raw.rolling(window).mean()], axis=1).dropna()
        if aligned.empty:
            return _result("rolling_window_shift", True, "insufficient_non_null_rows", "warn")
        unshifted_match = np.allclose(aligned.iloc[:, 0], aligned.iloc[:, 1], equal_nan=False)
        return _result(
            "rolling_window_shift",
            not bool(unshifted_match),
            "ok" if not unshifted_match else "feature_matches_unshifted_rolling_value",
        )

    def check_scaler_fit_scope(
        self,
        fit_indices: Iterable[object],
        train_indices: Iterable[object],
    ) -> LeakageCheckResult:
        fit_set = set(fit_indices)
        train_set = set(train_indices)
        leaked = sorted(str(index) for index in fit_set.difference(train_set))
        return _result(
            "scaler_fit_scope",
            not leaked,
            "ok" if not leaked else "processor_fit_used_non_train_rows",
            details={"leaked_indices": leaked},
        )

    def check_purge_embargo(
        self,
        train_end_position: int,
        test_start_position: int,
        purge_bars: int,
        embargo_bars: int,
    ) -> LeakageCheckResult:
        required_gap = max(int(purge_bars), 0) + max(int(embargo_bars), 0)
        actual_gap = int(test_start_position) - int(train_end_position) - 1
        passed = actual_gap >= required_gap
        return _result(
            "purge_embargo",
            passed,
            "ok" if passed else "insufficient_purge_embargo_gap",
            details={"required_gap": required_gap, "actual_gap": actual_gap},
        )
