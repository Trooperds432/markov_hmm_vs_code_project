from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .execution.paper_state import (
    append_ledger,
    load_paper_state,
    position_snapshot_rows,
    resolve_paper_path,
    save_paper_state,
)
from .execution.risk_checks import run_risk_checks


def run_paper_simulation(
    cfg: PipelineConfig,
    backtest: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    output_dir = Path("outputs") if output_dir is None else Path(output_dir)
    latest = backtest.iloc[-1]
    target_weight = float(latest.get("position", 0.0))
    timestamp = pd.Timestamp(latest.name).isoformat()
    trade_date = pd.Timestamp(latest.name).date().isoformat()
    price = float(latest["Close"])
    state, state_path = load_paper_state(cfg, output_dir)
    positions = dict(state.get("positions", {}))
    current_position = dict(positions.get(cfg.ticker, {}))
    current_qty = float(current_position.get("quantity", 0.0) or 0.0)
    current_market_value = current_qty * price
    cash = float(state.get("cash", cfg.paper_initial_cash) or 0.0)
    equity = cash + sum(
        float(pos.get("quantity", 0.0) or 0.0) * float(pos.get("last_price", price) or price)
        for pos in positions.values()
    )
    if equity <= 0.0:
        equity = float(cfg.paper_initial_cash)
    current_weight = current_market_value / equity if equity else 0.0
    target_notional = target_weight * equity
    delta_notional = target_notional - current_market_value
    if not cfg.paper_allow_fractional_shares:
        quantity_delta = float(np.trunc(delta_notional / price))
    else:
        quantity_delta = delta_notional / price if price else 0.0
    trade_notional = abs(quantity_delta * price)

    checks = run_risk_checks(cfg, latest, target_weight, equity)
    extra_checks = [
        {
            "check": "min_trade_notional",
            "passed": trade_notional >= cfg.paper_min_trade_notional or trade_notional == 0.0,
            "value": trade_notional,
            "limit": cfg.paper_min_trade_notional,
            "current_equity": equity,
        },
        {
            "check": "rebalance_threshold",
            "passed": abs(target_weight - current_weight) >= cfg.paper_rebalance_threshold,
            "value": abs(target_weight - current_weight),
            "limit": cfg.paper_rebalance_threshold,
            "current_equity": equity,
        },
        {
            "check": "no_duplicate_trade_same_day",
            "passed": True,
            "value": trade_date,
            "limit": "one trade per ticker per day",
            "current_equity": equity,
        },
    ]
    trade_ledger_path = resolve_paper_path(
        cfg.paper_trade_ledger_path,
        output_dir,
        "paper_trade_ledger.csv",
    )
    if trade_ledger_path.exists() and trade_ledger_path.stat().st_size > 0:
        ledger = pd.read_csv(trade_ledger_path)
        if {"date", "ticker"}.issubset(ledger.columns):
            duplicate = ledger["date"].astype(str).eq(trade_date) & ledger["ticker"].astype(str).eq(
                cfg.ticker
            )
            extra_checks[-1]["passed"] = not bool(duplicate.any())
    checks = pd.concat([checks, pd.DataFrame(extra_checks)], ignore_index=True)
    orders = []
    fills = []
    if checks["passed"].all():
        fill_price = price * (1.0 + np.sign(quantity_delta) * cfg.paper_slippage_bps / 10_000.0)
        gross = quantity_delta * fill_price
        commission = abs(gross) * cfg.paper_commission_bps / 10_000.0
        cash -= gross + commission
        new_qty = current_qty + quantity_delta
        if abs(new_qty) < 1e-12:
            positions.pop(cfg.ticker, None)
        else:
            old_cost = current_qty * float(current_position.get("avg_price", price) or price)
            added_cost = quantity_delta * fill_price
            avg_price = (old_cost + added_cost) / new_qty if new_qty else 0.0
            market_value = new_qty * price
            positions[cfg.ticker] = {
                "quantity": float(new_qty),
                "avg_price": float(avg_price),
                "last_price": price,
                "market_value": float(market_value),
                "weight": float(market_value / equity if equity else 0.0),
            }
        orders.append(
            {
                "timestamp": timestamp,
                "date": trade_date,
                "ticker": cfg.ticker,
                "target_weight": target_weight,
                "action": "buy" if quantity_delta > 0 else "sell",
                "quantity_estimate": quantity_delta,
                "reason": "filled",
            }
        )
        fills.append(
            {
                "timestamp": timestamp,
                "date": trade_date,
                "ticker": cfg.ticker,
                "action": orders[-1]["action"],
                "fill_price": fill_price,
                "fill_qty": quantity_delta,
                "commission": commission,
                "slippage": abs(quantity_delta * price) * cfg.paper_slippage_bps / 10_000.0,
            }
        )
    else:
        orders.append(
            {
                "timestamp": timestamp,
                "date": trade_date,
                "ticker": cfg.ticker,
                "target_weight": target_weight,
                "action": "blocked",
                "quantity_estimate": 0.0,
                "reason": "risk_check_failed",
            }
        )

    for ticker, position in positions.items():
        last_price = price if ticker == cfg.ticker else float(position.get("last_price", price))
        quantity = float(position.get("quantity", 0.0) or 0.0)
        position["last_price"] = last_price
        position["market_value"] = quantity * last_price
    equity = cash + sum(float(pos.get("market_value", 0.0) or 0.0) for pos in positions.values())
    for position in positions.values():
        position["weight"] = (
            float(position.get("market_value", 0.0) or 0.0) / equity if equity else 0.0
        )
    state.update(
        {
            "cash": float(cash),
            "positions": positions,
            "unrealized_pnl": float(
                sum(
                    (float(pos.get("last_price", 0.0)) - float(pos.get("avg_price", 0.0)))
                    * float(pos.get("quantity", 0.0))
                    for pos in positions.values()
                )
            ),
            "equity": float(equity),
            "last_update": trade_date,
        }
    )
    save_paper_state(state, state_path)

    ledger_rows = [
        {
            **fill,
            "notional": float(fill["fill_price"] * fill["fill_qty"]),
            "cash_after": float(cash),
            "equity_after": float(equity),
        }
        for fill in fills
    ]
    if ledger_rows:
        trade_ledger = append_ledger(trade_ledger_path, ledger_rows)
    elif trade_ledger_path.exists() and trade_ledger_path.stat().st_size > 0:
        trade_ledger = pd.read_csv(trade_ledger_path)
    else:
        trade_ledger = pd.DataFrame(
            columns=[
                "timestamp",
                "date",
                "ticker",
                "action",
                "fill_price",
                "fill_qty",
                "commission",
                "slippage",
                "notional",
                "cash_after",
                "equity_after",
            ]
        )
    position_history_path = output_dir / "paper_position_history.csv"
    cash_ledger_path = output_dir / "paper_cash_ledger.csv"
    append_ledger(position_history_path, position_snapshot_rows(state, timestamp))
    append_ledger(
        cash_ledger_path,
        [
            {
                "timestamp": timestamp,
                "date": trade_date,
                "cash": float(cash),
                "equity": float(equity),
                "realized_pnl": float(state.get("realized_pnl", 0.0) or 0.0),
                "unrealized_pnl": float(state.get("unrealized_pnl", 0.0) or 0.0),
            }
        ],
    )
    daily_equity = pd.DataFrame(
        [
            {
                "date": trade_date,
                "equity": equity,
                "cash": cash,
                "gross_exposure": abs(target_weight),
                "net_exposure": target_weight,
                "drawdown": 0.0,
            }
        ]
    )
    return {
        "paper_orders": pd.DataFrame(orders),
        "paper_fills": pd.DataFrame(fills),
        "paper_positions": pd.DataFrame(position_snapshot_rows(state, timestamp)),
        "paper_daily_equity": daily_equity,
        "paper_equity": daily_equity,
        "paper_risk_checks": checks,
        "paper_order_status": pd.DataFrame(
            [
                {
                    "order_id": idx,
                    "status": order["reason"],
                    "ticker": order["ticker"],
                    "quantity": order["quantity_estimate"],
                }
                for idx, order in enumerate(orders, start=1)
            ]
        ),
        "paper_trade_ledger": trade_ledger,
    }
