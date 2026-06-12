from __future__ import annotations

import numpy as np
import pandas as pd


def date_indexed(
    frame: pd.DataFrame, date_columns: tuple[str, ...] = ("date", "Date", "ds")
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    for column in date_columns:
        if column in out:
            out[column] = pd.to_datetime(out[column])
            return out.set_index(column).sort_index()
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def asof_to_base(
    features: pd.DataFrame, base_index: pd.Index, limit: int | None = None
) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame(index=base_index)
    left = pd.DataFrame({"date": pd.to_datetime(base_index)})
    right = features.sort_index().copy()
    right.index = pd.to_datetime(right.index)
    right = right.reset_index().rename(columns={right.index.name or "index": "date"})
    joined = pd.merge_asof(left, right.sort_values("date"), on="date", direction="backward")
    out = joined.set_index("date").reindex(pd.to_datetime(base_index))
    out.index = base_index
    if limit is not None:
        out = out.ffill(limit=limit)
    return out


def safe_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean().shift(1)
    std = series.rolling(window).std(ddof=0).shift(1)
    return (series - mean) / std.replace(0.0, np.nan)


def safe_log_return(frame: pd.DataFrame, column: str, periods: int) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index)
    values = np.log(frame[column].astype(float) / frame[column].astype(float).shift(periods))
    return values.replace([np.inf, -np.inf], np.nan)
