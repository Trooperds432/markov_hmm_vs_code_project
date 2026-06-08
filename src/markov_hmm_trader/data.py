from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from .config import PipelineConfig


def download_ohlcv(symbol: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
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


def _coverage_row(
    symbol: str, name: str, data: pd.DataFrame | None, error: str = ""
) -> dict[str, object]:
    if data is None or data.empty:
        return {
            "symbol": symbol,
            "name": name,
            "first_date": "",
            "last_date": "",
            "rows": 0,
            "missing_ratio": np.nan,
            "error": error,
        }

    close = data["Close"] if "Close" in data else pd.Series(dtype=float)
    return {
        "symbol": symbol,
        "name": name,
        "first_date": data.index.min().date().isoformat(),
        "last_date": data.index.max().date().isoformat(),
        "rows": int(len(data)),
        "missing_ratio": float(close.isna().mean()) if len(close) else np.nan,
        "error": error,
    }


def load_market_data(cfg: PipelineConfig) -> pd.DataFrame:
    asset = download_ohlcv(cfg.ticker, cfg.start, cfg.end, cfg.interval)
    bench = download_ohlcv(cfg.benchmark, cfg.start, cfg.end, cfg.interval)
    coverage_rows = [
        _coverage_row(cfg.ticker, "asset", asset),
        _coverage_row(cfg.benchmark, "benchmark", bench),
    ]

    df = asset.join(
        bench[["Close"]].rename(columns={"Close": "Benchmark_Close"}),
        how="inner",
    )

    if cfg.use_extra_factor_features:
        for name, symbol in cfg.extra_symbols.items():
            try:
                extra = download_ohlcv(symbol, cfg.start, cfg.end, cfg.interval)
                coverage_rows.append(_coverage_row(symbol, name, extra))
            except Exception as err:
                coverage_rows.append(_coverage_row(symbol, name, None, error=type(err).__name__))
                if not cfg.allow_missing_extra_symbols:
                    raise
                continue

            column = f"{name}_Close"
            df = df.join(extra[["Close"]].rename(columns={"Close": column}), how="left")
            df[column] = df[column].ffill(limit=max(int(cfg.factor_ffill_limit), 0))

    if df.empty:
        raise ValueError(
            "Joined asset/benchmark data is empty. Check ticker, benchmark, and date range."
        )

    if cfg.use_extra_factor_features:
        required = ["Open", "High", "Low", "Close", "Volume", "Benchmark_Close"]
        out = df.dropna(subset=required).copy()
    else:
        out = df.dropna().copy()
    out.attrs["factor_data_coverage"] = coverage_rows
    return out


def factor_data_coverage(df: pd.DataFrame) -> pd.DataFrame:
    coverage = df.attrs.get("factor_data_coverage")
    if isinstance(coverage, pd.DataFrame):
        return coverage.copy()
    if isinstance(coverage, list):
        return pd.DataFrame(coverage)
    return pd.DataFrame()
