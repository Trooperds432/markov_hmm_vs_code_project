from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PipelineConfig
from .base import DataProviderResult, UnsupportedCapabilityMixin, metadata_frame


class CSVProvider(UnsupportedCapabilityMixin):
    name = "csv"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def supports(self, capability: str) -> bool:
        return capability == "price_history"

    def _path_for(self, symbol: str) -> Path:
        if symbol == self.cfg.benchmark and self.cfg.csv_benchmark_path:
            return Path(self.cfg.csv_benchmark_path)
        if self.cfg.csv_price_path:
            return Path(self.cfg.csv_price_path)
        if self.cfg.csv_data_dir:
            return Path(self.cfg.csv_data_dir) / f"{symbol}.csv"
        return Path(self.cfg.data_cache_dir) / f"{symbol}.csv"

    def get_price_history(
        self,
        symbol: str,
        start: str,
        end: str | None,
        interval: str,
    ) -> DataProviderResult:
        path = self._path_for(symbol)
        data = pd.read_csv(path)
        date_column = "Date" if "Date" in data.columns else data.columns[0]
        data[date_column] = pd.to_datetime(data[date_column])
        data = data.set_index(date_column).sort_index()
        if start:
            data = data.loc[pd.Timestamp(start) :]
        if end:
            data = data.loc[: pd.Timestamp(end)]
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [column for column in required if column not in data.columns]
        if missing:
            raise ValueError(f"CSV provider missing columns for {symbol}: {missing}")
        out = data[required].copy()
        return DataProviderResult(out, metadata=metadata_frame(self.name, symbol, out))
