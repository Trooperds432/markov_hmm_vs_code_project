from __future__ import annotations

from src.markov_hmm_trader.config import (
    PipelineConfig,
    apply_profile,
    apply_quick_mode,
    config_hash,
    load_config,
)


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


def test_stage2_alias_config_fields_are_mapped(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        """
        {
          "observable_markov_window": 11,
          "observable_markov_threshold": 0.03,
          "observable_markov_forecast_horizons": [1, 3],
          "hmm_model_type_candidates": ["gaussian", "gmm"],
          "vol_rank_window": 55,
          "overfit_train_validation_gap_warn": 0.4,
          "overfit_validation_test_gap_warn": 0.2
        }
        """,
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.observable_label_window == 11
    assert cfg.observable_bull_threshold == 0.03
    assert cfg.observable_bear_threshold == -0.03
    assert cfg.observable_forecast_steps == (1, 3)
    assert cfg.hmm_model_types == ("gaussian", "gmm")
    assert cfg.vol_risk_lookback == 55
    assert cfg.overfit_train_validation_gap_threshold == 0.4
    assert cfg.overfit_validation_test_gap_threshold == 0.2


def test_stage2e_feature_group_aliases_and_tuple_fields_load(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        """
        {
          "feature_groups_enabled": ["price_core", "qlib_alpha"],
          "hmm_feature_set": ["z_ret_1"],
          "feature_winsorize_limits": [0.02, 0.98]
        }
        """,
        encoding="utf-8",
    )

    cfg = load_config(path)

    assert cfg.feature_groups_enabled == ("price_core", "alpha158_lite")
    assert cfg.hmm_feature_set == ("z_ret_1",)
    assert cfg.feature_winsorize_limits == (0.02, 0.98)
