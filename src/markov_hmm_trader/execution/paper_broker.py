from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

from .broker_base import BrokerBase


@dataclass
class PaperBroker(BrokerBase):
    initial_cash: float = 100000.0
    slippage_bps: float = 0.0
    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)
    orders: dict[str, dict[str, object]] = field(default_factory=dict)
    _ids: count = field(default_factory=lambda: count(1), init=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)

    def get_cash(self) -> float:
        return float(self.cash)

    def get_positions(self) -> dict[str, float]:
        return dict(self.positions)

    def submit_order(self, symbol: str, quantity: float, price: float) -> dict[str, object]:
        order_id = f"PAPER-{next(self._ids)}"
        fill_price = float(price) * (1.0 + self.slippage_bps / 10_000.0 * (1 if quantity > 0 else -1))
        notional = float(quantity) * fill_price
        self.cash -= notional
        self.positions[symbol] = self.positions.get(symbol, 0.0) + float(quantity)
        order = {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": float(quantity),
            "price": float(price),
            "fill_price": fill_price,
            "status": "filled",
        }
        self.orders[order_id] = order
        return order

    def cancel_order(self, order_id: str) -> dict[str, object]:
        order = self.orders.get(order_id, {"order_id": order_id, "status": "not_found"})
        if order.get("status") == "open":
            order["status"] = "cancelled"
        return dict(order)

    def get_order_status(self, order_id: str) -> dict[str, object]:
        return dict(self.orders.get(order_id, {"order_id": order_id, "status": "not_found"}))

    def equity(self, prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, quantity in self.positions.items():
            value += quantity * float(prices.get(symbol, 0.0))
        return float(value)
