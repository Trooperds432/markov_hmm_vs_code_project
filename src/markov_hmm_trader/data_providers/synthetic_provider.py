from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from .base import DataProviderResult, UnsupportedCapabilityMixin, metadata_frame


class SyntheticProvider(UnsupportedCapabilityMixin):
    name = "synthetic"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg

    def supports(self, capability: str) -> bool:
        return capability in {"price_history", "breadth"}

    def get_price_history(
        self,
        symbol: str,
        start: str,
        end: str | None,
        interval: str,
    ) -> DataProviderResult:
        periods = max(int(self.cfg.synthetic_periods), 80)
        rng = np.random.default_rng(int(self.cfg.random_seed) + sum(ord(ch) for ch in symbol))
        index = pd.bdate_range(start or "2020-01-01", periods=periods)
        drift = 0.0003 + 0.0002 * np.sin(np.linspace(0, 8 * np.pi, periods))
        returns = drift + rng.normal(0.0, 0.012, periods)
        close = 100.0 * np.exp(np.cumsum(returns))
        open_ = close * (1.0 + rng.normal(0.0, 0.002, periods))
        high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.002, 0.003, periods)))
        low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.002, 0.003, periods)))
        volume = rng.integers(1_000_000, 5_000_000, periods)
        data = pd.DataFrame(
            {
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=index,
        )
        return DataProviderResult(data=data, metadata=metadata_frame(self.name, symbol, data))

    def get_market_snapshots(
        self, symbols: list[str] | None = None, *args, **kwargs
    ) -> DataProviderResult:
        symbols = symbols or list(self.cfg.universe[:5])
        rng = np.random.default_rng(int(self.cfg.random_seed))
        data = pd.DataFrame(
            {
                "symbol": symbols,
                "return_1d": rng.normal(0.0, 0.02, len(symbols)),
                "return_20d": rng.normal(0.01, 0.08, len(symbols)),
                "vol_20d": rng.uniform(0.10, 0.45, len(symbols)),
                "sector": ["synthetic"] * len(symbols),
            }
        )
        return DataProviderResult(data=data)
