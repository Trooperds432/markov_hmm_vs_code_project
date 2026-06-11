from __future__ import annotations

from ..config import PipelineConfig
from .base import BaseDataProvider
from .csv_provider import CSVProvider
from .openbb_provider import OpenBBProvider
from .synthetic_provider import SyntheticProvider
from .yfinance_provider import YFinanceProvider


def get_data_provider(cfg: PipelineConfig) -> BaseDataProvider:
    name = cfg.data_provider.lower()
    if name == "yfinance":
        return YFinanceProvider(cfg)
    if name == "csv":
        return CSVProvider(cfg)
    if name == "synthetic":
        return SyntheticProvider(cfg)
    if name == "openbb":
        return OpenBBProvider(cfg)
    choices = ", ".join(cfg.data_provider_candidates)
    raise ValueError(f"Unknown data_provider={cfg.data_provider!r}. Valid providers: {choices}.")
