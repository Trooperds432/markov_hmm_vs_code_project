from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .leakage import FeatureLeakageGuard, LeakageCheckResult


class BaseProcessor(Protocol):
    def fit(self, df: pd.DataFrame) -> BaseProcessor: ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame: ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class ProcessorChain:
    processors: list[BaseProcessor] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> ProcessorChain:
        current = df.copy()
        for processor in self.processors:
            processor.fit(current)
            current = processor.transform(current)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        current = df.copy()
        for processor in self.processors:
            current = processor.transform(current)
        return current

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)


@dataclass
class DropMissingProcessor:
    columns: list[str] | None = None

    def fit(self, df: pd.DataFrame) -> DropMissingProcessor:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(subset=self.columns).copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class FillMissingProcessor:
    value: float | None = None
    method: str = "ffill"
    limit: int | None = None

    def fit(self, df: pd.DataFrame) -> FillMissingProcessor:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.value is not None:
            return df.fillna(self.value)
        if self.method == "ffill":
            return df.ffill(limit=self.limit)
        if self.method == "bfill":
            return df.bfill(limit=self.limit)
        raise ValueError(f"Unsupported fill method: {self.method}")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class WinsorizeProcessor:
    columns: list[str] | None = None
    lower_quantile: float = 0.01
    upper_quantile: float = 0.99
    bounds_: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> WinsorizeProcessor:
        columns = self.columns or list(df.select_dtypes(include=[np.number]).columns)
        self.bounds_ = {}
        for column in columns:
            series = df[column].dropna()
            if series.empty:
                continue
            self.bounds_[column] = (
                float(series.quantile(self.lower_quantile)),
                float(series.quantile(self.upper_quantile)),
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for column, (lower, upper) in self.bounds_.items():
            if column in out:
                out[column] = out[column].clip(lower=lower, upper=upper)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class ZScoreProcessor:
    columns: list[str] | None = None
    means_: dict[str, float] = field(default_factory=dict)
    stds_: dict[str, float] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> ZScoreProcessor:
        columns = self.columns or list(df.select_dtypes(include=[np.number]).columns)
        self.means_ = {column: float(df[column].mean()) for column in columns if column in df}
        self.stds_ = {
            column: float(df[column].std(ddof=0)) or 1.0 for column in columns if column in df
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for column, mean in self.means_.items():
            std = self.stds_.get(column, 1.0) or 1.0
            out[column] = (out[column] - mean) / std
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class RollingZScoreProcessor:
    columns: list[str]
    window: int = 60
    shift: int = 1

    def fit(self, df: pd.DataFrame) -> RollingZScoreProcessor:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for column in self.columns:
            mean = out[column].rolling(self.window).mean().shift(self.shift)
            std = out[column].rolling(self.window).std(ddof=0).shift(self.shift)
            out[f"{column}_rz{self.window}"] = (out[column] - mean) / std.replace(0.0, np.nan)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class FeatureColumnFilter:
    columns: list[str]

    def fit(self, df: pd.DataFrame) -> FeatureColumnFilter:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        existing = [column for column in self.columns if column in df]
        return df[existing].copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class RegimeLabelProcessor:
    return_column: str = "ret_20"
    bull_threshold: float = 0.02
    bear_threshold: float = -0.02
    output_column: str = "observable_regime"

    def fit(self, df: pd.DataFrame) -> RegimeLabelProcessor:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.output_column] = np.select(
            [
                out[self.return_column] >= self.bull_threshold,
                out[self.return_column] <= self.bear_threshold,
            ],
            ["Bull", "Bear"],
            default="Sideways",
        )
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class AsOfJoinProcessor:
    right: pd.DataFrame
    tolerance: pd.Timedelta | None = None
    suffix: str = ""

    def fit(self, df: pd.DataFrame) -> AsOfJoinProcessor:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        left = df.sort_index().copy()
        right = self.right.sort_index().copy()
        joined = pd.merge_asof(
            left.reset_index().rename(columns={"index": "date"}),
            right.reset_index().rename(columns={"index": "date"}),
            on="date",
            direction="backward",
            tolerance=self.tolerance,
            suffixes=("", self.suffix),
        ).set_index("date")
        joined.index.name = df.index.name
        return joined

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class FrequencyAlignmentProcessor:
    limit: int | None = None

    def fit(self, df: pd.DataFrame) -> FrequencyAlignmentProcessor:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_index().ffill(limit=self.limit)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


@dataclass
class LeakageGuardProcessor:
    feature_columns: list[str]
    guard: FeatureLeakageGuard = field(default_factory=FeatureLeakageGuard)
    results_: list[LeakageCheckResult] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> LeakageGuardProcessor:
        self.results_ = [self.guard.check_no_future_columns(self.feature_columns)]
        failures = [result for result in self.results_ if result.status == "fail"]
        if failures:
            raise ValueError("; ".join(result.message for result in failures))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
