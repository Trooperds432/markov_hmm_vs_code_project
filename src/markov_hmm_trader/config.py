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
    data_provider: str = "yfinance"
    data_provider_candidates: tuple[str, ...] = ("yfinance", "openbb", "csv", "synthetic")
    data_cache_dir: str = "cache"
    data_failover_to_cache: bool = True
    csv_data_dir: str | None = None
    csv_price_path: str = ""
    csv_benchmark_path: str = ""
    synthetic_periods: int = 360
    feature_groups_enabled: tuple[str, ...] = ("price_core", "volatility_core")
    optional_feature_groups: tuple[str, ...] = (
        "technical",
        "qlib_alpha",
        "alpha158_lite",
        "breadth",
        "macro",
        "fundamentals",
        "news",
        "options",
        "famafrench",
        "alt_bars",
    )
    hmm_feature_set: tuple[str, ...] | None = None
    max_feature_missing_ratio: float = 0.30
    feature_winsorize_limits: tuple[float, float] = (0.01, 0.99)
    use_processor_pipeline: bool = True
    enable_research_public_features: bool = False
    research_public_feature_set: tuple[str, ...] = (
        "stationarity",
        "autoregressive",
        "volatility_clustering",
        "market_residual",
        "correlation_regime",
        "regression_diagnostics",
        "pca_market_mode",
        "cointegration_spread",
        "factor_ranks",
        "instability",
    )
    research_public_windows: tuple[int, ...] = (20, 60, 120)
    stationarity_windows: tuple[int, ...] = (60, 120)
    acf_lags: tuple[int, ...] = (1, 5, 10)
    rolling_beta_window: int = 60
    rolling_corr_window: int = 60
    pca_window: int = 120
    pca_universe_symbols: tuple[str, ...] = ("SPY", "QQQ", "SMH", "XLK")
    cointegration_window: int = 252
    cointegration_peer_symbols: tuple[str, ...] = ()
    factor_rank_universe: tuple[str, ...] = ()
    hypothesis_alpha: float = 0.05
    min_periods_ratio: float = 0.70
    use_research_public_hmm_features: bool = False
    research_public_hmm_features: tuple[str, ...] = (
        "ret_1",
        "ret_5",
        "ret_20",
        "acf_1",
        "acf_5",
        "ar1_beta_60d",
        "ar1_residual_zscore",
        "ewma_vol_20d",
        "ewma_vol_60d",
        "vol_ratio_20_60",
        "vol_of_vol_20d",
        "tail_ratio_vs_normal",
        "adf_pvalue_60d",
        "rolling_mean_instability",
        "rolling_vol_instability",
        "rolling_beta_instability",
        "rolling_beta_spy_60d",
        "market_model_r2",
        "market_residual_zscore",
        "corr_spy_60d",
        "average_market_correlation",
        "pc1_explained_variance",
        "asset_loading_pc1",
        "idiosyncratic_residual_from_pca",
        "sector_spread_zscore",
        "spread_half_life",
        "rank_percentile",
    )
    use_statistical_confidence_scaling: bool = False
    min_statistical_confidence_mult: float = 0.25
    max_statistical_confidence_mult: float = 1.25
    use_research_public_state_mapping: bool = False

    # HMM
    n_states: int = 3
    train_window: int = 756
    retrain_frequency: int = 21
    n_iter: int = 500
    tolerance: float = 1e-4
    covariance_type: str = "full"
    random_seeds: tuple[int, ...] = (7, 13, 21, 42, 99, 123, 777)
    enable_hmm_model_selection: bool = False
    hmm_model_type: str = "gaussian"
    hmm_model_types: tuple[str, ...] = ("gaussian",)
    hmm_model_type_candidates: tuple[str, ...] | None = None
    hmm_state_candidates: tuple[int, ...] = (3, 4, 5)
    hmm_covariance_candidates: tuple[str, ...] = ("diag", "full")
    hmm_n_mix: int = 2
    hmm_n_mix_candidates: tuple[int, ...] = (2,)
    hmm_min_covar: float = 1e-3
    hmm_min_covar_candidates: tuple[float, ...] = (1e-3,)
    hmm_startprob_prior: float = 1.0
    hmm_transmat_prior: float = 1.0
    hmm_transmat_prior_candidates: tuple[float, ...] = (1.0,)
    hmm_means_prior: float = 0.0
    hmm_means_weight: float = 0.0
    hmm_covars_prior: float = 0.01
    hmm_covars_weight: float = 1.0
    hmm_weights_prior: float = 1.0
    hmm_params: str = "stmc"
    hmm_init_params: str = "stmc"
    hmm_params_gmm: str = "stmcw"
    hmm_init_params_gmm: str = "stmcw"
    hmm_decode_algorithm: str = "viterbi"
    hmm_implementation: str = "scaling"
    hmm_use_native_information_criteria: bool = True
    hmm_store_convergence_history: bool = True
    hmm_model_selection_metric: str = "validation_bic"
    hmm_validation_fraction: float = 0.20
    refit_selected_hmm_on_full_train: bool = True
    semantic_min_occupancy: float = 0.03
    hmm_synthetic_sample_paths: int = 25
    hmm_synthetic_sample_length: int = 252

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
    random_seed: int = 42
    deterministic: bool = True
    purge_bars: int = 0
    embargo_bars: int = 0

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

    # Observable Markov baseline
    enable_observable_markov: bool = False
    observable_label_window: int = 20
    observable_bull_threshold: float = 0.02
    observable_bear_threshold: float = -0.02
    observable_forecast_steps: tuple[int, ...] = (1, 5, 20, 60)
    observable_markov_window: int | None = None
    observable_markov_threshold: float | None = None
    observable_markov_forecast_horizons: tuple[int, ...] | None = None

    # Forecast benchmarks and volatility forecasts
    enable_forecast_benchmarks: bool = False
    forecast_models: tuple[str, ...] = ("naive", "auto_arima", "auto_ets", "auto_theta")
    forecast_horizon_bars: int = 20
    forecast_cv_windows: int = 5
    forecast_step_size: int = 21
    forecast_freq: str = "B"
    forecast_levels: tuple[int, ...] = (80, 95)
    forecast_refit_every: int = 1
    enable_volatility_forecasts: bool = True
    vol_model: str = "realised_vol"
    vol_model_candidates: tuple[str, ...] = ("realised_vol", "ewma_vol", "garch", "arch")
    ewma_lambda: float = 0.94
    vol_forecast_horizon_bars: int = 20
    vol_forecast_refit_frequency: int = 21
    vol_rank_window: int | None = None
    garch_p: int = 1
    garch_q: int = 1
    arch_p: int = 1
    vol_risk_lookback: int = 252
    vol_min_risk_multiplier: float = 0.25
    vol_max_risk_multiplier: float = 1.0

    # Overfit diagnostics
    enable_overfit_report: bool = True
    overfit_train_validation_gap_threshold: float = 0.25
    overfit_validation_test_gap_threshold: float = 0.25
    overfit_train_validation_gap_warn: float | None = None
    overfit_validation_test_gap_warn: float | None = None

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
    nn_model_type: str = "mlp"
    nn_model_types: tuple[str, ...] = ("mlp", "conv_lstm")
    nn_conv_channels: int = 16
    nn_mc_dropout_samples: int = 20
    nn_uncertainty_penalty: float = 1.0
    nn_require_oos_metadata: bool = True
    nn_use_time_decay_weights: bool = True
    nn_time_decay_factor: float = 0.001
    nn_early_stopping_patience: int = 5
    nn_lr_patience: int = 2
    nn_lr_factor: float = 0.5
    nn_min_delta: float = 1e-4
    ensemble_mode: str = "hmm_only"
    hmm_weight: float = 0.65
    nn_weight: float = 0.35
    observable_markov_weight: float = 0.10
    forecast_weight: float = 0.10

    # Portfolio construction
    portfolio_enabled: bool = False
    portfolio_mode: str = "single_asset"
    portfolio_symbols: tuple[str, ...] = ("NVDA", "MSFT", "AAPL", "META", "AMZN", "AMD", "TSLA")
    portfolio_benchmark: str = "QQQ"
    portfolio_max_positions: int = 5
    portfolio_max_weight_per_asset: float = 0.25
    portfolio_cash_buffer: float = 0.05
    portfolio_target_vol: float = 0.20
    portfolio_rebalance_frequency: str = "monthly"
    portfolio_weighting_method: str = "inverse_vol"
    portfolio_sector_cap: float = 0.50

    # Experiment tracking
    enable_experiment_tracking: bool = True
    experiment_name: str = "default_hmm_research"
    experiment_root: str = "experiments/runs"
    save_run_manifest: bool = True
    enable_model_registry: bool = True
    model_registry_path: str = "experiments/registry/model_registry.csv"
    registry_require_oos_pass: bool = True

    # Paper trading simulation
    enable_paper_trading: bool = False
    paper_initial_cash: float = 100000.0
    paper_rebalance_frequency: str = "D"
    paper_execution_lag_bars: int = 1
    paper_commission_bps: float = 2.0
    paper_slippage_bps: float = 2.0
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

    if "universe_tickers" in raw and "universe" not in raw:
        raw["universe"] = raw["universe_tickers"]
    if isinstance(raw.get("portfolio_mode"), bool):
        raw["portfolio_enabled"] = bool(raw["portfolio_mode"])
        raw["portfolio_mode"] = "top_k" if raw["portfolio_enabled"] else "single_asset"
    if "hmm_model_type_candidates" in raw and "hmm_model_types" not in raw:
        raw["hmm_model_types"] = raw["hmm_model_type_candidates"]
    if "observable_markov_window" in raw and "observable_label_window" not in raw:
        raw["observable_label_window"] = raw["observable_markov_window"]
    if "observable_markov_threshold" in raw:
        if "observable_bull_threshold" not in raw:
            raw["observable_bull_threshold"] = raw["observable_markov_threshold"]
        if "observable_bear_threshold" not in raw:
            raw["observable_bear_threshold"] = -float(raw["observable_markov_threshold"])
    if "observable_markov_forecast_horizons" in raw and "observable_forecast_steps" not in raw:
        raw["observable_forecast_steps"] = raw["observable_markov_forecast_horizons"]
    if "vol_rank_window" in raw and "vol_risk_lookback" not in raw:
        raw["vol_risk_lookback"] = raw["vol_rank_window"]
    if (
        "overfit_train_validation_gap_warn" in raw
        and "overfit_train_validation_gap_threshold" not in raw
    ):
        raw["overfit_train_validation_gap_threshold"] = raw["overfit_train_validation_gap_warn"]
    if (
        "overfit_validation_test_gap_warn" in raw
        and "overfit_validation_test_gap_threshold" not in raw
    ):
        raw["overfit_validation_test_gap_threshold"] = raw["overfit_validation_test_gap_warn"]
    for key in (
        "feature_groups_enabled",
        "optional_feature_groups",
        "hmm_feature_set",
        "research_public_feature_set",
        "research_public_hmm_features",
    ):
        if key in raw and isinstance(raw[key], list):
            raw[key] = ["alpha158_lite" if item == "qlib_alpha" else item for item in raw[key]]

    valid_names = {field.name for field in fields(PipelineConfig)}
    clean = {k: v for k, v in raw.items() if k in valid_names}

    tuple_fields = {
        "random_seeds",
        "data_provider_candidates",
        "feature_groups_enabled",
        "optional_feature_groups",
        "hmm_feature_set",
        "research_public_feature_set",
        "research_public_windows",
        "stationarity_windows",
        "acf_lags",
        "pca_universe_symbols",
        "cointegration_peer_symbols",
        "factor_rank_universe",
        "research_public_hmm_features",
        "feature_winsorize_limits",
        "hmm_model_types",
        "hmm_model_type_candidates",
        "hmm_state_candidates",
        "hmm_covariance_candidates",
        "hmm_n_mix_candidates",
        "hmm_min_covar_candidates",
        "hmm_transmat_prior_candidates",
        "stress_cost_multipliers",
        "stress_execution_lag_bars",
        "stress_commission_bps_values",
        "stress_slippage_bps_values",
        "stress_missing_data_probabilities",
        "universe",
        "observable_forecast_steps",
        "observable_markov_forecast_horizons",
        "forecast_models",
        "forecast_levels",
        "vol_model_candidates",
        "nn_hidden_sizes",
        "nn_model_types",
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
        forecast_cv_windows=min(cfg.forecast_cv_windows, 2),
        forecast_step_size=max(cfg.forecast_step_size, cfg.forecast_horizon_bars),
        vol_forecast_refit_frequency=max(cfg.vol_forecast_refit_frequency, 42),
        nested_trials_per_window=min(cfg.nested_trials_per_window, 5),
        nn_epochs=min(cfg.nn_epochs, 3),
        nn_mc_dropout_samples=min(cfg.nn_mc_dropout_samples, 5),
    )


def config_to_dict(cfg: PipelineConfig) -> dict[str, Any]:
    return asdict(cfg)


def config_hash(cfg: PipelineConfig) -> str:
    payload = json.dumps(config_to_dict(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
