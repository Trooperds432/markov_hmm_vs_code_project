from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig

KNOWN_DATA_PROVIDERS = {"yfinance", "csv", "synthetic", "openbb"}
KNOWN_PORTFOLIO_MODES = {"single_asset", "top_k", "risk_budgeted"}
KNOWN_PORTFOLIO_WEIGHTING = {
    "equal",
    "signal_weighted",
    "score",
    "inverse_vol",
    "vol_inverse",
    "risk_budgeted",
    "risk_parity_simple",
    "score_vol_inverse",
}
KNOWN_ENSEMBLE_MODES = {
    "hmm_only",
    "nn_only",
    "hmm_and_nn_agree",
    "weighted_blend",
    "multi_signal",
}
KNOWN_NN_MODEL_TYPES = {"mlp", "conv_lstm"}
KNOWN_FEATURE_GROUPS = {
    "price_core",
    "volatility_core",
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
    "research_public",
    "glassdoor_sentiment",
    "wsb_attention",
    "valuation_anchor",
    "alt_data_overlays",
}


@dataclass(frozen=True)
class ConfigValidationIssue:
    field: str
    severity: str
    message: str
    value: object | None = None


def _issue(
    issues: list[ConfigValidationIssue],
    field: str,
    severity: str,
    message: str,
    value: object | None,
) -> None:
    issues.append(ConfigValidationIssue(field, severity, message, value))


def validate_config(cfg: PipelineConfig) -> list[ConfigValidationIssue]:
    issues: list[ConfigValidationIssue] = []
    if cfg.n_states < 2:
        _issue(issues, "n_states", "error", "must be >= 2", cfg.n_states)
    if cfg.data_provider not in KNOWN_DATA_PROVIDERS:
        _issue(
            issues,
            "data_provider",
            "error",
            f"must be one of {sorted(KNOWN_DATA_PROVIDERS)}",
            cfg.data_provider,
        )
    provider_candidates = set(cfg.data_provider_candidates)
    if cfg.data_provider not in provider_candidates:
        _issue(
            issues,
            "data_provider",
            "error",
            "must be included in data_provider_candidates",
            cfg.data_provider,
        )
    if not isinstance(cfg.portfolio_mode, str) or cfg.portfolio_mode not in KNOWN_PORTFOLIO_MODES:
        _issue(
            issues,
            "portfolio_mode",
            "error",
            f"must be one of {sorted(KNOWN_PORTFOLIO_MODES)}",
            cfg.portfolio_mode,
        )
    if cfg.portfolio_weighting_method not in KNOWN_PORTFOLIO_WEIGHTING:
        _issue(
            issues,
            "portfolio_weighting_method",
            "error",
            f"must be one of {sorted(KNOWN_PORTFOLIO_WEIGHTING)}",
            cfg.portfolio_weighting_method,
        )
    if cfg.ensemble_mode not in KNOWN_ENSEMBLE_MODES:
        _issue(
            issues,
            "ensemble_mode",
            "error",
            f"must be one of {sorted(KNOWN_ENSEMBLE_MODES)}",
            cfg.ensemble_mode,
        )
    if cfg.nn_model_type not in KNOWN_NN_MODEL_TYPES:
        _issue(
            issues,
            "nn_model_type",
            "error",
            f"must be one of {sorted(KNOWN_NN_MODEL_TYPES)}",
            cfg.nn_model_type,
        )
    unknown_groups = sorted(set(cfg.feature_groups_enabled).difference(KNOWN_FEATURE_GROUPS))
    if unknown_groups:
        _issue(
            issues,
            "feature_groups_enabled",
            "error",
            f"unknown feature groups: {unknown_groups}",
            unknown_groups,
        )
    if cfg.train_window <= cfg.retrain_frequency:
        _issue(
            issues,
            "train_window",
            "warning",
            "train_window should exceed retrain_frequency/test window",
            cfg.train_window,
        )
    if cfg.vol_lookback >= cfg.train_window:
        _issue(
            issues, "vol_lookback", "warning", "should be less than train_window", cfg.vol_lookback
        )
    if not 0.0 < cfg.portfolio_max_weight_per_asset <= 1.0:
        _issue(
            issues,
            "portfolio_max_weight_per_asset",
            "error",
            "must be > 0 and <= 1",
            cfg.portfolio_max_weight_per_asset,
        )
    if not 0.0 <= cfg.portfolio_cash_buffer <= 1.0:
        _issue(
            issues,
            "portfolio_cash_buffer",
            "error",
            "must be between 0 and 1",
            cfg.portfolio_cash_buffer,
        )
    if cfg.paper_max_position_weight > cfg.paper_max_portfolio_leverage:
        _issue(
            issues,
            "paper_max_position_weight",
            "error",
            "must be <= paper_max_portfolio_leverage",
            cfg.paper_max_position_weight,
        )
    if (
        cfg.enable_paper_trading
        and cfg.paper_model_registry_required
        and not Path(cfg.model_registry_path).exists()
    ):
        _issue(
            issues,
            "paper_model_registry_required",
            "warning",
            "model registry is required for paper mode but registry file is missing",
            cfg.model_registry_path,
        )
    if cfg.max_exposure > 1.0:
        _issue(
            issues,
            "max_exposure",
            "warning",
            "max_exposure > 1 enables leverage; confirm this is intentional",
            cfg.max_exposure,
        )
    if cfg.long_only and cfg.max_exposure < 0.0:
        _issue(
            issues,
            "max_exposure",
            "error",
            "must be non-negative in long-only mode",
            cfg.max_exposure,
        )
    if "options" in cfg.feature_groups_enabled and cfg.data_provider == "yfinance":
        _issue(
            issues,
            "feature_groups_enabled",
            "warning",
            "provider_lacks_options",
            "options",
        )
    if (
        any(
            group in cfg.feature_groups_enabled
            for group in {"options", "macro", "fundamentals", "news"}
        )
        and cfg.data_provider == "openbb"
        and importlib.util.find_spec("openbb") is None
    ):
        _issue(
            issues,
            "data_provider",
            "warning",
            "OpenBB feature groups requested but openbb is missing",
            cfg.data_provider,
        )
    if cfg.use_alt_data_hmm_features and not any(
        [cfg.glassdoor_reviews_path, cfg.wsb_mentions_path, cfg.valuation_data_path]
    ):
        _issue(
            issues,
            "use_alt_data_hmm_features",
            "warning",
            "alt-data HMM features requested but no local alt-data paths are configured",
            True,
        )
    if cfg.enable_nn_confirmation and importlib.util.find_spec("torch") is None:
        _issue(
            issues,
            "enable_nn_confirmation",
            "warning",
            "torch is missing; NN confirmation will be skipped",
            True,
        )
    if cfg.enable_forecast_benchmarks and importlib.util.find_spec("statsforecast") is None:
        _issue(
            issues,
            "enable_forecast_benchmarks",
            "warning",
            "statsforecast is missing; forecast benchmarks will use fallback/skips",
            True,
        )
    if cfg.use_research_public_hmm_features and not cfg.enable_research_public_features:
        _issue(
            issues,
            "use_research_public_hmm_features",
            "warning",
            "research_public HMM features requested while research_public feature block is disabled",
            True,
        )
    if cfg.feature_corr_drop_threshold is not None and not (
        0.0 < cfg.feature_corr_drop_threshold <= 1.0
    ):
        _issue(
            issues,
            "feature_corr_drop_threshold",
            "error",
            "must be None or in (0, 1]",
            cfg.feature_corr_drop_threshold,
        )
    if not 0.0 < cfg.feature_corr_warn_threshold <= 1.0:
        _issue(
            issues,
            "feature_corr_warn_threshold",
            "error",
            "must be in (0, 1]",
            cfg.feature_corr_warn_threshold,
        )
    if cfg.min_signal < 0.0:
        _issue(issues, "min_signal", "error", "must be >= 0", cfg.min_signal)
    elif cfg.min_signal < 1e-9:
        _issue(
            issues,
            "min_signal",
            "warning",
            "near-zero min_signal weakens confidence scaling",
            cfg.min_signal,
        )
    if not isinstance(cfg.random_seed, int):
        _issue(issues, "random_seed", "error", "must be an int", cfg.random_seed)
    if not isinstance(cfg.deterministic, bool):
        _issue(issues, "deterministic", "error", "must be a bool", cfg.deterministic)
    if cfg.purge_bars < 0:
        _issue(issues, "purge_bars", "error", "must be >= 0", cfg.purge_bars)
    if cfg.embargo_bars < 0:
        _issue(issues, "embargo_bars", "error", "must be >= 0", cfg.embargo_bars)
    if cfg.commission_bps < 0.0:
        _issue(issues, "commission_bps", "error", "must be >= 0", cfg.commission_bps)
    if cfg.slippage_bps < 0.0:
        _issue(issues, "slippage_bps", "error", "must be >= 0", cfg.slippage_bps)
    return issues


def assert_config_valid(cfg: PipelineConfig) -> None:
    errors = [issue for issue in validate_config(cfg) if issue.severity == "error"]
    if errors:
        message = "; ".join(f"{issue.field}: {issue.message}" for issue in errors)
        raise ValueError(f"Invalid configuration: {message}")
