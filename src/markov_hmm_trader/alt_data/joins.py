from __future__ import annotations

import numpy as np
import pandas as pd


def safe_asof_join(
    market: pd.DataFrame,
    alt: pd.DataFrame,
    ticker: str,
    date_col: str,
    lag_days: int,
    max_age_days: int | None = None,
) -> pd.DataFrame:
    if alt.empty or date_col not in alt or "ticker" not in alt:
        return pd.DataFrame(index=market.index)
    market_dates = pd.DataFrame({"market_date": pd.to_datetime(market.index).tz_localize(None)})
    alt_local = alt.copy()
    alt_local = alt_local[alt_local["ticker"].astype(str).str.upper().eq(str(ticker).upper())]
    if alt_local.empty:
        return pd.DataFrame(index=market.index)
    alt_local[date_col] = pd.to_datetime(alt_local[date_col], errors="coerce").dt.tz_localize(None)
    alt_local["available_date"] = alt_local[date_col] + pd.to_timedelta(int(lag_days), unit="D")
    alt_local = alt_local.dropna(subset=["available_date"]).sort_values("available_date")
    joined = pd.merge_asof(
        market_dates.sort_values("market_date"),
        alt_local,
        left_on="market_date",
        right_on="available_date",
        direction="backward",
    ).set_index(market.index)
    if date_col in joined:
        age = (joined["market_date"] - joined[date_col]).dt.days
        joined[f"{date_col}_age_days"] = age
        if max_age_days is not None:
            stale = age > int(max_age_days)
            joined.loc[
                stale, joined.columns.difference(["market_date", f"{date_col}_age_days"])
            ] = np.nan
    return joined.drop(columns=["market_date"], errors="ignore")
