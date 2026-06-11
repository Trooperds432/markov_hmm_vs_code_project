from __future__ import annotations

import importlib.util

import pandas as pd

from ..config import PipelineConfig
from .base import DataProviderResult, UnsupportedCapabilityMixin, metadata_frame


class OpenBBProvider(UnsupportedCapabilityMixin):
    name = "openbb"

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.available = importlib.util.find_spec("openbb") is not None

    def supports(self, capability: str) -> bool:
        if not self.available:
            return False
        return capability in {
            "price_history",
            "options",
            "options_surface",
            "macro",
            "fundamentals",
            "news",
            "breadth",
            "famafrench",
        }

    def _missing(self, capability: str) -> DataProviderResult:
        return DataProviderResult(
            pd.DataFrame(),
            warnings=pd.DataFrame(
                [
                    {
                        "provider": self.name,
                        "capability": capability,
                        "reason": "missing_optional_dependency",
                        "detail": "openbb",
                    }
                ]
            ),
        )

    def get_price_history(
        self,
        symbol: str,
        start: str,
        end: str | None,
        interval: str,
    ) -> DataProviderResult:
        if not self.available:
            return self._missing("price_history")
        try:
            from openbb import obb

            frame = obb.equity.price.historical(
                symbol=symbol,
                start_date=start,
                end_date=end,
                interval=interval,
            ).to_df()
            rename = {
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            frame = frame.rename(columns=rename)
            if "date" in frame:
                frame["date"] = pd.to_datetime(frame["date"])
                frame = frame.set_index("date")
            required = ["Open", "High", "Low", "Close", "Volume"]
            out = frame[required].dropna().copy()
            return DataProviderResult(out, metadata=metadata_frame(self.name, symbol, out))
        except Exception as err:
            warnings = pd.DataFrame(
                [
                    {
                        "provider": self.name,
                        "capability": "price_history",
                        "reason": "provider_error",
                        "detail": type(err).__name__,
                    }
                ]
            )
            return DataProviderResult(pd.DataFrame(), warnings=warnings)
