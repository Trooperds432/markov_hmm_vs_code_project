from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from .config import PipelineConfig


def download_ohlcv(symbol: str, start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    data = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if data.empty:
        raise ValueError(f"No data returned for {symbol}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns for {symbol}: {missing}")

    data = data[["Open", "High", "Low", "Close", "Volume"]].copy()
    data.index = pd.to_datetime(data.index)
    return data.dropna()


def load_market_data(cfg: PipelineConfig) -> pd.DataFrame:
    asset = download_ohlcv(cfg.ticker, cfg.start, cfg.end, cfg.interval)
    bench = download_ohlcv(cfg.benchmark, cfg.start, cfg.end, cfg.interval)

    df = asset.join(
        bench[["Close"]].rename(columns={"Close": "Benchmark_Close"}),
        how="inner",
    )

    if df.empty:
        raise ValueError("Joined asset/benchmark data is empty. Check ticker, benchmark, and date range.")

    return df.dropna()
