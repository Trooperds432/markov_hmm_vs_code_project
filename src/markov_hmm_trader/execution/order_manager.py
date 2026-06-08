from __future__ import annotations

import pandas as pd

from .broker_base import BrokerBase


def target_weight_order(
    broker: BrokerBase,
    symbol: str,
    target_weight: float,
    price: float,
    equity: float,
) -> dict[str, object]:
    current_quantity = broker.get_positions().get(symbol, 0.0)
    target_quantity = float(target_weight) * float(equity) / float(price)
    quantity_delta = target_quantity - current_quantity
    if abs(quantity_delta) < 1e-9:
        return {
            "order_id": "",
            "symbol": symbol,
            "quantity": 0.0,
            "price": float(price),
            "status": "no_order",
        }
    return broker.submit_order(symbol, quantity_delta, price)


def orders_frame(orders: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(orders)
