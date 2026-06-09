from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.execution.order_manager import target_weight_order
from src.markov_hmm_trader.execution.paper_broker import PaperBroker
from src.markov_hmm_trader.execution.risk_checks import run_risk_checks
from src.markov_hmm_trader.paper_trading import run_paper_simulation


def test_paper_order_lifecycle():
    broker = PaperBroker(initial_cash=1000.0)

    order = target_weight_order(broker, "SPY", 0.5, 100.0, 1000.0)

    assert order["status"] == "filled"
    assert broker.get_positions()["SPY"] == 5.0
    assert broker.get_order_status(order["order_id"])["status"] == "filled"


def test_unapproved_model_blocks_paper_trading(tmp_path):
    latest = pd.Series({"Bull": 0.8, "Bear": 0.1, "ret_1": 0.0})
    cfg = PipelineConfig(ticker="SPY", paper_model_registry_required=True)

    checks = run_risk_checks(
        cfg,
        latest,
        target_weight=0.1,
        current_equity=1000.0,
        registry_path=tmp_path / "missing_registry.csv",
    )

    approval = checks.set_index("check").loc["no_trade_if_unapproved_model"]
    assert bool(approval["passed"]) is False


def test_paper_simulation_emits_csv_artifact_frames():
    dates = pd.bdate_range("2020-01-01", periods=2)
    backtest = pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "position": [0.0, 0.1],
            "Bull": [0.7, 0.8],
            "Bear": [0.1, 0.1],
            "ret_1": [0.0, 0.01],
        },
        index=dates,
    )
    cfg = PipelineConfig(paper_model_registry_required=False)

    outputs = run_paper_simulation(cfg, backtest)

    assert {"paper_orders", "paper_fills", "paper_positions", "paper_daily_equity"}.issubset(
        outputs
    )
    assert "target_weight" in outputs["paper_orders"].columns
