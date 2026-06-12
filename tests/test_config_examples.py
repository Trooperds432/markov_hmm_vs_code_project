from __future__ import annotations

from pathlib import Path

import pytest

from src.markov_hmm_trader.config import load_config
from src.markov_hmm_trader.config_validation import validate_config


@pytest.mark.parametrize("path", sorted(Path("configs").glob("*.json")))
def test_all_top_level_configs_load(path: Path):
    cfg = load_config(path)

    assert cfg.ticker
    assert cfg.portfolio_mode in {"single_asset", "top_k", "risk_budgeted"}


def test_default_config_uses_canonical_portfolio_fields():
    text = Path("configs/default.json").read_text(encoding="utf-8")
    cfg = load_config("configs/default.json")

    assert '"portfolio_mode": false' not in text
    assert cfg.portfolio_enabled is False
    assert cfg.portfolio_mode == "single_asset"


def test_invalid_config_values_are_errors():
    cfg = load_config("configs/basic_hmm.json")
    cfg.portfolio_mode = "bad"  # type: ignore[misc]
    cfg.ensemble_mode = "bad"  # type: ignore[misc]

    errors = [issue for issue in validate_config(cfg) if issue.severity == "error"]

    assert {issue.field for issue in errors} >= {"portfolio_mode", "ensemble_mode"}
