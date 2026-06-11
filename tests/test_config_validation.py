from __future__ import annotations

import pytest

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.config_validation import assert_config_valid, validate_config


def test_default_config_validation_has_no_errors():
    issues = validate_config(PipelineConfig())

    assert not [issue for issue in issues if issue.severity == "error"]


def test_invalid_config_values_are_detected():
    cfg = PipelineConfig(
        n_states=1,
        portfolio_max_weight_per_asset=1.5,
        portfolio_cash_buffer=-0.1,
        commission_bps=-1,
        slippage_bps=-1,
    )

    issues = validate_config(cfg)

    assert {issue.field for issue in issues if issue.severity == "error"} >= {
        "n_states",
        "portfolio_max_weight_per_asset",
        "portfolio_cash_buffer",
        "commission_bps",
        "slippage_bps",
    }
    with pytest.raises(ValueError):
        assert_config_valid(cfg)


def test_dangerous_near_zero_signal_is_warning():
    issues = validate_config(PipelineConfig(min_signal=0.0))

    assert any(issue.field == "min_signal" and issue.severity == "warning" for issue in issues)
