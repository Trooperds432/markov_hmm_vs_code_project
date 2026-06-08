from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any


@dataclass
class PipelineConfig:
    # Market data
    ticker: str = "NVDA"
    benchmark: str = "QQQ"
    start: str = "2012-01-01"
    end: str | None = None
    interval: str = "1d"

    # HMM
    n_states: int = 3
    train_window: int = 756
    retrain_frequency: int = 21
    n_iter: int = 500
    tolerance: float = 1e-4
    covariance_type: str = "full"
    random_seeds: tuple[int, ...] = (7, 13, 21, 42, 99, 123, 777)
    enable_hmm_model_selection: bool = False
    hmm_state_candidates: tuple[int, ...] = (3, 4, 5)
    hmm_covariance_candidates: tuple[str, ...] = ("diag", "full")
    hmm_model_selection_metric: str = "validation_bic"
    hmm_validation_fraction: float = 0.20
    refit_selected_hmm_on_full_train: bool = True
    semantic_min_occupancy: float = 0.03

    # Signal thresholds
    long_only: bool = True
    min_bull_prob: float = 0.42
    max_bear_prob_for_long: float = 0.45
    min_signal: float = 0.05
    min_alpha_score: int = 4

    # Exposure sizing
    base_exposure: float = 0.35
    core_exposure: float = 0.55
    max_exposure: float = 0.95
    bear_regime_exposure_mult: float = 0.20
    side_regime_exposure_mult: float = 0.60
    bull_regime_exposure_mult: float = 1.00

    # Volatility targeting
    use_vol_targeting: bool = True
    target_ann_vol: float = 0.28
    vol_lookback: int = 30
    min_vol_mult: float = 0.35
    max_vol_mult: float = 1.75

    # Drawdown throttle
    use_drawdown_throttle: bool = True
    dd_level_1: float = 0.10
    dd_level_2: float = 0.22
    dd_mult_1: float = 0.70
    dd_mult_2: float = 0.40

    # Costs
    commission_bps: float = 7.5
    slippage_bps: float = 3.0

    # Trend and relative strength
    fast_ema: int = 50
    slow_ema: int = 200
    rs_fast: int = 20
    rs_slow: int = 60

    # Reporting
    bars_per_year: int = 252

    # Diagnostics and stress testing
    enable_diagnostics: bool = True
    enable_stress_tests: bool = True
    enable_benchmark: bool = False
    stress_cost_multipliers: tuple[float, ...] = (2.0, 3.0)
    stress_execution_lag_bars: tuple[int, ...] = (1, 2)
    stress_bootstrap_paths: int = 200
    stress_bootstrap_block_size: int = 21
    stress_random_seed: int = 1234
    stress_asset_bootstrap_paths: int = 50
    stress_asset_bootstrap_block_size: int = 21
    stress_rerun_hmm_on_bootstrap: bool = True
    stress_commission_bps_values: tuple[float, ...] = (7.5, 15.0, 30.0)
    stress_slippage_bps_values: tuple[float, ...] = (3.0, 10.0, 25.0)
    stress_missing_data_probabilities: tuple[float, ...] = (0.01, 0.03, 0.05)
    custom_crisis_windows: dict[str, dict[str, str]] = field(default_factory=dict)
    benchmark_maxiter: int = 200

    # Universe and factor data
    universe: tuple[str, ...] = (
        "SPY",
        "QQQ",
        "SMH",
        "XLK",
        "NVDA",
        "MSFT",
        "AAPL",
        "META",
        "AMZN",
        "AMD",
        "TSLA",
    )
    extra_symbols: dict[str, str] = field(
        default_factory=lambda: {
            "market": "SPY",
            "tech": "QQQ",
            "semiconductors": "SMH",
            "volatility_proxy": "^VIX",
            "rates_proxy": "TLT",
            "dollar_proxy": "UUP",
            "credit_proxy": "HYG",
            "defensive_proxy": "XLU",
            "growth_proxy": "VUG",
            "value_proxy": "VTV",
        }
    )
    allow_missing_extra_symbols: bool = True
    use_extra_factor_features: bool = False
    factor_ffill_limit: int = 5

    # Nested walk-forward optimisation
    enable_nested_optimisation: bool = False
    outer_train_years: int = 5
    outer_validation_years: int = 1
    outer_test_years: int = 1
    outer_step_years: int = 1
    nested_trials_per_window: int = 30
    nested_objective: str = "balanced"

    # Optional neural-network confirmation
    enable_nn_confirmation: bool = False
    nn_horizon_bars: int = 20
    nn_lookback_bars: int = 60
    nn_hidden_sizes: tuple[int, ...] = (64, 32)
    nn_dropout: float = 0.10
    nn_learning_rate: float = 0.001
    nn_epochs: int = 20
    nn_batch_size: int = 64
    nn_validation_split: float = 0.2
    nn_positive_threshold: float = 0.55
    nn_negative_threshold: float = 0.45
    ensemble_mode: str = "hmm_only"
    hmm_weight: float = 0.65
    nn_weight: float = 0.35

    # Portfolio construction
    portfolio_mode: bool = False
    portfolio_symbols: tuple[str, ...] = ("NVDA", "MSFT", "AAPL", "META", "AMZN", "AMD", "TSLA")
    portfolio_benchmark: str = "QQQ"
    portfolio_max_positions: int = 5
    portfolio_max_weight_per_asset: float = 0.25
    portfolio_target_vol: float = 0.20
    portfolio_rebalance_frequency: str = "monthly"
    portfolio_weighting_method: str = "inverse_vol"

    # Experiment tracking
    enable_experiment_tracking: bool = True
    experiment_name: str = "default_hmm_research"

    # Paper trading simulation
    enable_paper_trading: bool = False
    paper_initial_cash: float = 100000.0
    paper_max_daily_loss: float = 0.03
    paper_max_drawdown_stop: float = 0.15
    paper_model_registry_required: bool = True
    paper_max_position_weight: float = 0.25
    paper_max_portfolio_leverage: float = 1.0
    paper_min_model_confidence: float = 0.55
    paper_max_data_staleness_days: int = 5


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    valid_names = {field.name for field in fields(PipelineConfig)}
    clean = {k: v for k, v in raw.items() if k in valid_names}

    tuple_fields = {
        "random_seeds",
        "hmm_state_candidates",
        "hmm_covariance_candidates",
        "stress_cost_multipliers",
        "stress_execution_lag_bars",
        "stress_commission_bps_values",
        "stress_slippage_bps_values",
        "stress_missing_data_probabilities",
        "universe",
        "nn_hidden_sizes",
        "portfolio_symbols",
    }
    for name in tuple_fields:
        if name in clean and isinstance(clean[name], list):
            clean[name] = tuple(clean[name])

    return PipelineConfig(**clean)


PROFILE_OVERRIDES: dict[str, dict[str, Any]] = {
    "balanced": {
        "min_bull_prob": 0.42,
        "max_bear_prob_for_long": 0.45,
        "min_signal": 0.05,
        "base_exposure": 0.35,
        "core_exposure": 0.55,
        "max_exposure": 0.95,
        "target_ann_vol": 0.28,
        "dd_level_1": 0.10,
        "dd_level_2": 0.22,
        "dd_mult_1": 0.70,
        "dd_mult_2": 0.40,
        "bear_regime_exposure_mult": 0.20,
        "side_regime_exposure_mult": 0.60,
        "bull_regime_exposure_mult": 1.00,
    },
    "aggressive_growth": {
        "min_bull_prob": 0.35,
        "max_bear_prob_for_long": 0.65,
        "min_signal": 0.00,
        "base_exposure": 0.75,
        "core_exposure": 1.00,
        "max_exposure": 1.00,
        "target_ann_vol": 0.45,
        "dd_level_1": 0.22,
        "dd_level_2": 0.40,
        "dd_mult_1": 0.85,
        "dd_mult_2": 0.60,
        "bear_regime_exposure_mult": 0.45,
        "side_regime_exposure_mult": 0.90,
        "bull_regime_exposure_mult": 1.00,
    },
    "low_drawdown": {
        "min_bull_prob": 0.55,
        "max_bear_prob_for_long": 0.30,
        "min_signal": 0.12,
        "base_exposure": 0.25,
        "core_exposure": 0.40,
        "max_exposure": 0.65,
        "target_ann_vol": 0.16,
        "dd_level_1": 0.07,
        "dd_level_2": 0.15,
        "dd_mult_1": 0.55,
        "dd_mult_2": 0.25,
        "bear_regime_exposure_mult": 0.05,
        "side_regime_exposure_mult": 0.35,
        "bull_regime_exposure_mult": 0.90,
    },
}


def apply_profile(cfg: PipelineConfig, profile: str | None) -> PipelineConfig:
    if profile is None:
        return cfg
    if profile not in PROFILE_OVERRIDES:
        choices = ", ".join(sorted(PROFILE_OVERRIDES))
        raise ValueError(f"Unknown profile '{profile}'. Valid profiles: {choices}.")
    return replace(cfg, **PROFILE_OVERRIDES[profile])


def apply_overrides(cfg: PipelineConfig, **overrides: Any) -> PipelineConfig:
    valid_names = {field_item.name for field_item in fields(PipelineConfig)}
    clean = {k: v for k, v in overrides.items() if k in valid_names and v is not None}
    return replace(cfg, **clean)


def apply_quick_mode(cfg: PipelineConfig) -> PipelineConfig:
    quick_train_window = min(cfg.train_window, 252)
    return replace(
        cfg,
        train_window=max(60, quick_train_window),
        retrain_frequency=max(cfg.retrain_frequency, 42),
        n_iter=min(cfg.n_iter, 50),
        random_seeds=tuple(cfg.random_seeds[:2]) or (7,),
        stress_bootstrap_paths=min(cfg.stress_bootstrap_paths, 25),
        stress_asset_bootstrap_paths=min(cfg.stress_asset_bootstrap_paths, 5),
        hmm_state_candidates=tuple(cfg.hmm_state_candidates[:1]) or (cfg.n_states,),
        hmm_covariance_candidates=tuple(cfg.hmm_covariance_candidates[:1])
        or (cfg.covariance_type,),
        nested_trials_per_window=min(cfg.nested_trials_per_window, 5),
        nn_epochs=min(cfg.nn_epochs, 3),
    )


def config_to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    return asdict(cfg)


def config_hash(cfg: PipelineConfig) -> str:
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
