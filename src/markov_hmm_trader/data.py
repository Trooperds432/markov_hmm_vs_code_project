from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .data_providers.factory import get_data_provider
from .warnings import normalize_warning_frame


def download_ohlcv(symbol: str, start: str, end: str | None, interval: str) -> pd.DataFrame:
    import yfinance as yf

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
    provider = get_data_provider(cfg)
    asset_result = provider.get_price_history(cfg.ticker, cfg.start, cfg.end, cfg.interval)
    bench_result = provider.get_price_history(cfg.benchmark, cfg.start, cfg.end, cfg.interval)
    if asset_result.data.empty:
        raise ValueError(f"No data returned for {cfg.ticker} from provider {provider.name}.")
    if bench_result.data.empty:
        raise ValueError(f"No data returned for {cfg.benchmark} from provider {provider.name}.")
    asset = asset_result.data
    bench = bench_result.data
    coverage_rows = [
        _coverage_row(cfg.ticker, "asset", asset),
        _coverage_row(cfg.benchmark, "benchmark", bench),
    ]
    provider_warnings = []
    for warning_frame in (asset_result.warnings, bench_result.warnings):
        if not warning_frame.empty:
            provider_warnings.extend(warning_frame.to_dict("records"))

    df = asset.join(
        bench[["Close"]].rename(columns={"Close": "Benchmark_Close"}),
        how="inner",
    )

    if cfg.use_extra_factor_features:
        for name, symbol in cfg.extra_symbols.items():
            try:
                extra_result = provider.get_price_history(symbol, cfg.start, cfg.end, cfg.interval)
                extra = extra_result.data
                if not extra_result.warnings.empty:
                    provider_warnings.extend(extra_result.warnings.to_dict("records"))
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
    provider_metadata = pd.concat(
        [asset_result.metadata, bench_result.metadata],
        ignore_index=True,
    )
    out.attrs["data_provider_metadata"] = provider_metadata.to_dict("records")
    out.attrs["data_provider_warnings"] = provider_warnings
    return out


def factor_data_coverage(df: pd.DataFrame) -> pd.DataFrame:
    coverage = df.attrs.get("factor_data_coverage")
    if isinstance(coverage, pd.DataFrame):
        return coverage.copy()
    if isinstance(coverage, list):
        return pd.DataFrame(coverage)
    return pd.DataFrame()


def data_provider_warnings(df: pd.DataFrame) -> pd.DataFrame:
    columns = ["provider", "symbol", "capability", "reason", "detail"]
    warnings = df.attrs.get("data_provider_warnings")
    if isinstance(warnings, pd.DataFrame):
        frame = warnings.copy()
        for column in columns:
            if column not in frame:
                frame[column] = pd.Series(dtype=object)
        return normalize_warning_frame(frame).reindex(columns=columns)
    if isinstance(warnings, list):
        frame = pd.DataFrame(warnings)
        for column in columns:
            if column not in frame:
                frame[column] = pd.Series(dtype=object)
        return normalize_warning_frame(frame).reindex(columns=columns)
    return pd.DataFrame(columns=columns)


def all_data_provider_warnings(df: pd.DataFrame) -> pd.DataFrame:
    warnings = df.attrs.get("data_provider_warnings")
    if isinstance(warnings, pd.DataFrame):
        return normalize_warning_frame(warnings)
    if isinstance(warnings, list):
        return normalize_warning_frame(pd.DataFrame(warnings))
    return normalize_warning_frame(None)


def data_provider_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata = df.attrs.get("data_provider_metadata")
    if isinstance(metadata, pd.DataFrame):
        return metadata.copy()
    if isinstance(metadata, list):
        return pd.DataFrame(metadata)
    return pd.DataFrame(columns=["provider", "symbol", "capability", "status", "detail"])
