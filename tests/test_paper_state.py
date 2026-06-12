from __future__ import annotations

import json

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.paper_trading import run_paper_simulation


def _paper_backtest(weight: float = 0.2) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=2)
    return pd.DataFrame(
        {
            "Close": [100.0, 101.0],
            "position": [0.0, weight],
            "Bull": [0.7, 0.8],
            "Bear": [0.1, 0.1],
            "ret_1": [0.0, 0.01],
        },
        index=dates,
    )


def test_paper_state_persists_and_reset_works(tmp_path):
    cfg = PipelineConfig(
        ticker="SYNTH",
        paper_model_registry_required=False,
        paper_min_model_confidence=0.0,
        paper_rebalance_threshold=0.0,
    )

    run_paper_simulation(cfg, _paper_backtest(), output_dir=tmp_path)
    state_path = tmp_path / "paper_account_state.json"
    first_state = json.loads(state_path.read_text(encoding="utf-8"))
    run_paper_simulation(cfg, _paper_backtest(0.25), output_dir=tmp_path)
    ledger = pd.read_csv(tmp_path / "paper_trade_ledger.csv")

    assert first_state["equity"] > 0.0
    assert len(ledger) >= 1

    reset_cfg = PipelineConfig(
        ticker="SYNTH",
        paper_model_registry_required=False,
        paper_min_model_confidence=0.0,
        paper_reset_state=True,
    )
    run_paper_simulation(reset_cfg, _paper_backtest(0.0), output_dir=tmp_path)
    reset_state = json.loads(state_path.read_text(encoding="utf-8"))
    assert reset_state["cash"] == reset_cfg.paper_initial_cash
