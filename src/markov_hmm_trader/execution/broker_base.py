from __future__ import annotations

from abc import ABC, abstractmethod


class BrokerBase(ABC):
    @abstractmethod
    def get_cash(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, symbol: str, quantity: float, price: float) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, object]:
        raise NotImplementedError
