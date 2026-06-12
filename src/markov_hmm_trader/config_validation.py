from __future__ import annotations

from dataclasses import dataclass

from .config import PipelineConfig

KNOWN_DATA_PROVIDERS = {"yfinance", "csv", "synthetic", "openbb"}
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
