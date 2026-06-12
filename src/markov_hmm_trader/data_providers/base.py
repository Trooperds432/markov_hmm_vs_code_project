from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from ..warnings import StructuredWarning, warning_to_row


@dataclass(frozen=True)
class ProviderWarning:
    provider: str
    capability: str
    reason: str
    detail: str = ""


@dataclass
class DataProviderResult:
    data: pd.DataFrame
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: pd.DataFrame = field(default_factory=pd.DataFrame)


class BaseDataProvider(Protocol):
    name: str

    def supports(self, capability: str) -> bool: ...

    def get_price_history(
        self,
        symbol: str,
        start: str,
        end: str | None,
        interval: str,
    ) -> DataProviderResult: ...

    def get_company_news(
        self,
        symbol: str,
        start: str,
        end: str | None,
    ) -> DataProviderResult: ...

    def get_fundamentals(
        self,
        symbol: str,
        start: str,
        end: str | None,
    ) -> DataProviderResult: ...

    def get_options_chain(self, symbol: str, date: str | None = None) -> DataProviderResult: ...

    def get_options_surface(self, symbol: str, date: str | None = None) -> DataProviderResult: ...

    def get_macro_series(
        self,
        series_id: str,
        start: str,
        end: str | None,
    ) -> DataProviderResult: ...

    def get_market_snapshots(
        self,
        symbols: list[str] | None = None,
    ) -> DataProviderResult: ...

    def get_index_constituents(self, index_symbol: str) -> DataProviderResult: ...

    def get_fama_french(
        self,
        dataset: str,
        start: str,
        end: str | None,
    ) -> DataProviderResult: ...


class UnsupportedCapabilityMixin:
    name = "base"

    def supports(self, capability: str) -> bool:
        return capability == "price_history"

    def _unsupported(self, capability: str, reason: str = "unsupported_capability") -> pd.DataFrame:
        return pd.DataFrame(
            [
                warning_to_row(
                    StructuredWarning(
                        code=reason,
                        severity="warning",
                        module=f"data_provider.{self.name}",
                        message=f"{self.name} does not support {capability}.",
                        details={"provider": self.name, "capability": capability},
                    ),
                    provider=self.name,
                    capability=capability,
                    reason=reason,
                    detail=capability,
                )
            ]
        )

    def get_company_news(self, *args, **kwargs) -> DataProviderResult:
        return DataProviderResult(pd.DataFrame(), warnings=self._unsupported("news"))

    def get_fundamentals(self, *args, **kwargs) -> DataProviderResult:
        return DataProviderResult(pd.DataFrame(), warnings=self._unsupported("fundamentals"))

    def get_options_chain(self, *args, **kwargs) -> DataProviderResult:
        return DataProviderResult(pd.DataFrame(), warnings=self._unsupported("options"))

    def get_options_surface(self, *args, **kwargs) -> DataProviderResult:
        return DataProviderResult(pd.DataFrame(), warnings=self._unsupported("options_surface"))

    def get_macro_series(self, *args, **kwargs) -> DataProviderResult:
        return DataProviderResult(pd.DataFrame(), warnings=self._unsupported("macro"))

    def get_market_snapshots(self, *args, **kwargs) -> DataProviderResult:
        return DataProviderResult(pd.DataFrame(), warnings=self._unsupported("breadth"))

    def get_index_constituents(self, *args, **kwargs) -> DataProviderResult:
        return DataProviderResult(pd.DataFrame(), warnings=self._unsupported("constituents"))

    def get_fama_french(self, *args, **kwargs) -> DataProviderResult:
        return DataProviderResult(pd.DataFrame(), warnings=self._unsupported("famafrench"))


def metadata_frame(provider: str, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        first_date = ""
        last_date = ""
    else:
        first_date = data.index.min().date().isoformat()
        last_date = data.index.max().date().isoformat()
    return pd.DataFrame(
        [
            {
                "provider": provider,
                "symbol": symbol,
                "first_date": first_date,
                "last_date": last_date,
                "rows": int(len(data)),
            }
        ]
    )
