from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PipelineConfig
from .base import DataProviderResult, UnsupportedCapabilityMixin, metadata_frame


class YFinanceProvider(UnsupportedCapabilityMixin):
    name = "yfinance"

    def __init__(self, cfg: PipelineConfig | None = None):
        self.cfg = cfg

    def supports(self, capability: str) -> bool:
        return capability in {"price_history", "fundamentals", "news"}

    def _cache_path(self, symbol: str) -> Path:
        cache_dir = Path(self.cfg.data_cache_dir if self.cfg is not None else "cache")
        safe_symbol = symbol.replace("/", "_").replace("\\", "_").replace("^", "")
        return cache_dir / f"{safe_symbol}.csv"

    def _read_cache(self, symbol: str, start: str, end: str | None) -> DataProviderResult:
        path = self._cache_path(symbol)
        data = pd.read_csv(path)
        date_column = "Date" if "Date" in data.columns else data.columns[0]
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.set_index(date_column).sort_index()
        if start:
            data = data.loc[pd.Timestamp(start) :]
        if end:
            data = data.loc[: pd.Timestamp(end)]
        required = ["Open", "High", "Low", "Close", "Volume"]
        out = data[required].dropna().copy()
        warnings = pd.DataFrame(
            [
                {
                    "provider": self.name,
                    "symbol": symbol,
                    "capability": "price_history",
                    "reason": "used_cache",
                    "detail": str(path),
                }
            ]
        )
        return DataProviderResult(
            out, metadata=metadata_frame(self.name, symbol, out), warnings=warnings
        )

    def _maybe_read_cache(
        self,
        symbol: str,
        start: str,
        end: str | None,
        original_error: Exception | None = None,
    ) -> DataProviderResult | None:
        if self.cfg is None or not self.cfg.data_failover_to_cache:
            return None
        path = self._cache_path(symbol)
        if not path.exists():
            return None
        result = self._read_cache(symbol, start, end)
        if original_error is not None and not result.warnings.empty:
            result.warnings.loc[:, "reason"] = "live_provider_error_used_cache"
            result.warnings.loc[:, "detail"] = f"{type(original_error).__name__}; {path}"
        return result

    def _write_cache(self, symbol: str, data: pd.DataFrame) -> None:
        if self.cfg is None or not self.cfg.data_failover_to_cache:
            return
        path = self._cache_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.rename_axis("Date").to_csv(path)

    def get_price_history(
        self,
        symbol: str,
        start: str,
        end: str | None,
        interval: str,
    ) -> DataProviderResult:
        try:
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
        except Exception as err:
            cached = self._maybe_read_cache(symbol, start, end, original_error=err)
            if cached is not None:
                return cached
            raise
        if data.empty:
            cached = self._maybe_read_cache(symbol, start, end)
            if cached is not None:
                return cached
            raise ValueError(f"No data returned for {symbol}.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"Missing columns for {symbol}: {missing}")
        out = data[required].dropna().copy()
        out.index = pd.to_datetime(out.index)
        self._write_cache(symbol, out)
        return DataProviderResult(out, metadata=metadata_frame(self.name, symbol, out))
