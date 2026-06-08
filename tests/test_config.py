from __future__ import annotations

from src.markov_hmm_trader.config import PipelineConfig, apply_profile, apply_quick_mode, config_hash


def test_profiles_apply_distinct_risk_defaults():
    base = PipelineConfig()
    aggressive = apply_profile(base, "aggressive_growth")
    low_drawdown = apply_profile(base, "low_drawdown")

    assert aggressive.max_exposure > low_drawdown.max_exposure
    assert aggressive.target_ann_vol > low_drawdown.target_ann_vol
    assert low_drawdown.dd_mult_2 < aggressive.dd_mult_2


def test_quick_mode_reduces_expensive_settings():
    cfg = apply_quick_mode(PipelineConfig(random_seeds=(1, 2, 3), n_iter=500))

    assert cfg.n_iter == 50
    assert cfg.random_seeds == (1, 2)
    assert cfg.stress_bootstrap_paths <= 25


def test_config_hash_is_stable_for_same_config():
    cfg = PipelineConfig(ticker="SPY")

    assert config_hash(cfg) == config_hash(PipelineConfig(ticker="SPY"))
