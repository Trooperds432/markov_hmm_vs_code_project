from __future__ import annotations

import pandas as pd


def alt_data_feature_summary(features: pd.DataFrame) -> pd.DataFrame:
    prefixes = ("glassdoor_", "wsb_", "valuation_", "margin_of_safety", "price_to_intrinsic")
    rows = []
    for column in features.columns:
        if not column.startswith(prefixes):
            continue
        series = features[column]
        rows.append(
            {
                "feature": column,
                "missing_ratio": float(series.isna().mean()),
                "mean": float(pd.to_numeric(series, errors="coerce").mean()),
                "std": float(pd.to_numeric(series, errors="coerce").std(ddof=0)),
            }
        )
    return pd.DataFrame(rows)


def alt_data_warnings(*frames: pd.DataFrame) -> pd.DataFrame:
    clean = [frame for frame in frames if frame is not None and not frame.empty]
    return pd.concat(clean, ignore_index=True) if clean else pd.DataFrame()
