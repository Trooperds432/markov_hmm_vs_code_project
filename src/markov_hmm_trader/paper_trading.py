from __future__ import annotations

import pandas as pd

from .config import PipelineConfig
from .execution.order_manager import orders_frame, target_weight_order
from .execution.paper_broker import PaperBroker
from .execution.risk_checks import run_risk_checks


def run_paper_simulation(
    cfg: PipelineConfig,
    backtest: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    latest = backtest.iloc[-1]
    target_weight = float(latest.get("position", 0.0))
    broker = PaperBroker(
        initial_cash=cfg.paper_initial_cash,
        slippage_bps=cfg.paper_slippage_bps,
    )
    checks = run_risk_checks(cfg, latest, target_weight, broker.get_cash())
    orders = []
    fills = []
    if checks["passed"].all():
        order = target_weight_order(
            broker,
            cfg.ticker,
            target_weight,
            float(latest["Close"]),
            broker.get_cash(),
        )
        orders.append(
            {
                "timestamp": pd.Timestamp(latest.name).isoformat(),
                "ticker": cfg.ticker,
                "target_weight": target_weight,
                "action": "buy" if order.get("quantity", 0.0) > 0 else "sell",
                "quantity_estimate": order.get("quantity", 0.0),
                "reason": order.get("status", ""),
            }
        )
        fills.append(
            {
                "timestamp": pd.Timestamp(latest.name).isoformat(),
                "ticker": cfg.ticker,
                "action": orders[-1]["action"],
                "fill_price": order.get("fill_price", latest["Close"]),
                "fill_qty": order.get("quantity", 0.0),
                "commission": abs(float(order.get("quantity", 0.0)))
                * float(latest["Close"])
                * cfg.paper_commission_bps
                / 10_000.0,
                "slippage": abs(float(order.get("quantity", 0.0)))
                * float(latest["Close"])
                * cfg.paper_slippage_bps
                / 10_000.0,
            }
        )
    else:
        orders.append(
            {
                "timestamp": pd.Timestamp(latest.name).isoformat(),
                "ticker": cfg.ticker,
                "target_weight": target_weight,
                "action": "blocked",
                "quantity_estimate": 0.0,
                "reason": "risk_check_failed",
            }
        )

    equity = broker.equity({cfg.ticker: float(latest["Close"])})
    positions = broker.get_positions()
    return {
        "paper_orders": pd.DataFrame(orders),
        "paper_fills": pd.DataFrame(fills),
        "paper_positions": pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(latest.name).isoformat(),
                    "ticker": cfg.ticker,
                    "qty": positions.get(cfg.ticker, 0.0),
                    "market_value": positions.get(cfg.ticker, 0.0) * float(latest["Close"]),
                    "weight": (
                        positions.get(cfg.ticker, 0.0) * float(latest["Close"]) / equity
                        if equity
                        else 0.0
                    ),
                }
            ]
        ),
        "paper_daily_equity": pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(latest.name).date().isoformat(),
                    "equity": equity,
                    "cash": broker.get_cash(),
                    "gross_exposure": abs(target_weight),
                    "net_exposure": target_weight,
                    "drawdown": 0.0,
                }
            ]
        ),
        "paper_risk_checks": checks,
        "paper_order_status": orders_frame(
            [broker.get_order_status(order_id) for order_id in broker.orders]
        ),
    }
