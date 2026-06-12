from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig, config_hash
from .experiment import git_commit_hash

MODULE_OUTPUTS: dict[str, list[str]] = {
    "hmm": ["equity_curve.csv", "stats.csv", "feature_manifest.csv"],
    "hmm_model_selection": ["hmm_model_selection.csv"],
    "feature_quality": ["feature_quality_report.csv", "feature_drop_report.csv"],
    "research_public_features": [
        "research_public_feature_summary.csv",
        "factor_rank_diagnostics.csv",
        "long_short_spread_diagnostics.csv",
    ],
    "statistical_confidence_scaling": ["equity_curve.csv"],
    "observable_markov": [
        "observable_markov_baseline.csv",
        "observable_markov_transition_matrix.csv",
        "observable_markov_stationary_distribution.csv",
        "observable_markov_equity_curve.csv",
    ],
    "forecast_benchmarks": ["forecast_predictions.csv", "forecast_model_comparison.csv"],
    "volatility_forecasts": ["volatility_forecasts.csv"],
    "nn_confirmation": ["nn_predictions.csv", "nn_training_metrics.csv"],
    "ensemble": ["ensemble_diagnostics.csv"],
    "alt_data": [
        "glassdoor_features.csv",
        "wsb_attention_features.csv",
        "valuation_features.csv",
        "alt_data_overlay.csv",
    ],
    "portfolio": [
        "portfolio_equity_curve.csv",
        "portfolio_weights.csv",
        "portfolio_risk_report.csv",
        "portfolio_rebalance_decisions.csv",
        "portfolio_correlation_matrix.csv",
        "portfolio_constraint_report.csv",
    ],
    "paper": [
        "paper_orders.csv",
        "paper_account_state.json",
        "paper_trade_ledger.csv",
        "paper_position_history.csv",
        "paper_cash_ledger.csv",
        "paper_daily_equity.csv",
    ],
    "stress_tests": ["stress_tests.csv", "asset_bootstrap_stress.csv"],
    "random_baseline": [
        "random_baseline_summary.csv",
        "random_baseline_paths.csv",
        "random_timing_matched_exposure.csv",
        "random_timing_matched_turnover.csv",
        "random_rebalance_baseline.csv",
        "random_long_only_baseline.csv",
        "random_portfolio_baseline.csv",
    ],
    "hypothesis_validation": ["hypothesis_validation.csv", "statistical_validity_report.csv"],
    "experiment_tracking": ["experiment_manifest.json", "artifacts_index.json"],
    "output_audit": ["run_summary.json", "active_modules.csv", "output_status.csv"],
}

MODULE_DEPENDENCIES = {
    "forecast_benchmarks": "statsforecast",
    "nn_confirmation": "torch",
    "paper": "",
    "portfolio": "",
    "alt_data": "",
    "research_public_features": "statsmodels",
}


def _dependency_available(module: str) -> bool:
    dependency = MODULE_DEPENDENCIES.get(module, "")
    if not dependency:
        return True
    return importlib.util.find_spec(dependency) is not None


def module_enabled(cfg: PipelineConfig, module: str) -> bool:
    return {
        "hmm": True,
        "hmm_model_selection": cfg.enable_hmm_model_selection,
        "feature_quality": cfg.feature_quality_enabled,
        "research_public_features": cfg.enable_research_public_features,
        "statistical_confidence_scaling": cfg.use_statistical_confidence_scaling,
        "observable_markov": cfg.enable_observable_markov,
        "forecast_benchmarks": cfg.enable_forecast_benchmarks,
        "volatility_forecasts": cfg.enable_volatility_forecasts,
        "nn_confirmation": cfg.enable_nn_confirmation,
        "ensemble": cfg.ensemble_mode != "hmm_only",
        "alt_data": any(
            [
                cfg.enable_alt_data_features,
                cfg.enable_glassdoor_features,
                cfg.enable_wsb_features,
                cfg.enable_valuation_features,
                cfg.use_alt_data_position_overlay,
                cfg.use_alt_data_portfolio_ranking,
            ]
        ),
        "portfolio": cfg.portfolio_enabled,
        "paper": cfg.enable_paper_trading,
        "stress_tests": cfg.enable_stress_tests,
        "random_baseline": cfg.enable_random_baseline,
        "hypothesis_validation": cfg.enable_research_public_features,
        "experiment_tracking": cfg.enable_experiment_tracking,
        "output_audit": True,
    }.get(module, False)


def _csv_shape(path: Path) -> tuple[int, int]:
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeDecodeError):
        return 0, 0
    return int(len(frame)), int(len(frame.columns))


def output_status_frame(cfg: PipelineConfig, output_dir: Path) -> pd.DataFrame:
    rows = []
    expected: dict[str, str] = {}
    for module, files in MODULE_OUTPUTS.items():
        for file in files:
            expected[file] = module
    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            expected.setdefault(path.name, "other")

    for file, module in sorted(expected.items()):
        path = output_dir / file
        enabled = module_enabled(cfg, module) if module != "other" else True
        dep_ok = _dependency_available(module)
        rows_count = 0
        cols_count = 0
        reason = ""
        if not enabled:
            if path.exists() and path.suffix.lower() == ".csv":
                rows_count, cols_count = _csv_shape(path)
            status = "skipped_disabled"
            reason = "module_disabled"
        elif not dep_ok:
            status = "skipped_missing_dependency"
            reason = MODULE_DEPENDENCIES.get(module, "")
        elif path.exists() and path.stat().st_size > 0:
            if path.suffix.lower() == ".csv":
                rows_count, cols_count = _csv_shape(path)
                status = "active" if rows_count > 0 else "empty"
                reason = "" if rows_count > 0 else "file_has_header_but_no_rows"
            elif path.suffix.lower() == ".json":
                status = "active"
                rows_count = 1
                cols_count = 1
            else:
                status = "active"
                rows_count = 1
                cols_count = 1
        elif path.exists():
            status = "empty"
            reason = "zero_byte_file"
        else:
            status = "skipped_no_data"
            reason = "output_not_written"
        rows.append(
            {
                "file": file,
                "module": module,
                "enabled": bool(enabled),
                "active": status == "active",
                "status": status,
                "rows": rows_count,
                "columns": cols_count,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def active_modules_frame(cfg: PipelineConfig, status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for module, files in MODULE_OUTPUTS.items():
        enabled = module_enabled(cfg, module)
        dep_ok = _dependency_available(module)
        module_status = status[status["module"].eq(module)]
        active = bool(enabled and not module_status.empty and module_status["active"].any())
        if active:
            state = "active"
            reason = ""
        elif not enabled:
            state = "skipped_disabled"
            reason = "module_disabled"
        elif not dep_ok:
            state = "skipped_missing_dependency"
            reason = MODULE_DEPENDENCIES.get(module, "")
        elif module_status.empty:
            state = "skipped_no_data"
            reason = "no_expected_outputs"
        else:
            state = str(module_status["status"].iloc[0])
            reason = str(module_status["reason"].iloc[0])
        dependency = MODULE_DEPENDENCIES.get(module, "")
        rows.append(
            {
                "module": module,
                "enabled_by_config": bool(enabled),
                "active": active,
                "status": state,
                "reason": reason,
                "required_dependency": dependency,
                "dependency_available": dep_ok,
                "output_files": ",".join(files),
            }
        )
    return pd.DataFrame(rows)


def _warning_counts(output_dir: Path) -> dict[str, int]:
    warnings_path = output_dir / "warnings.csv"
    if not warnings_path.exists():
        return {"warning_count": 0, "error_count": 0}
    try:
        warnings = pd.read_csv(warnings_path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        return {"warning_count": 0, "error_count": 0}
    if "severity" in warnings:
        errors = int(warnings["severity"].astype(str).str.lower().eq("error").sum())
    else:
        errors = 0
    return {"warning_count": int(len(warnings)), "error_count": errors}


def write_run_audit_outputs(
    cfg: PipelineConfig,
    output_dir: Path,
    *,
    mode: str,
    raw_rows: int | None = None,
    feature_rows: int | None = None,
    backtest_rows: int | None = None,
    first_date: str = "",
    last_date: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    status = output_status_frame(cfg, output_dir)
    modules = active_modules_frame(cfg, status)

    feature_manifest_path = output_dir / "feature_manifest.csv"
    if feature_manifest_path.exists():
        manifest = pd.read_csv(feature_manifest_path)
        total_features = int(len(manifest))
        hmm_requested = int(manifest.get("used_in_hmm_requested", pd.Series(False)).sum())
        hmm_used = int(manifest.get("used_in_hmm", pd.Series(False)).sum())
        features_dropped = int(
            manifest.get("drop_reason", pd.Series(dtype=object)).fillna("").ne("").sum()
        )
    else:
        total_features = hmm_requested = hmm_used = features_dropped = 0

    warnings = _warning_counts(output_dir)
    active_files = status.loc[status["status"].eq("active"), "file"].tolist()
    empty_files = status.loc[status["status"].eq("empty"), "file"].tolist()
    skipped_files = status.loc[status["status"].str.startswith("skipped"), "file"].tolist()
    summary = {
        "run_id": config_hash(cfg)[:12],
        "ticker": cfg.ticker,
        "benchmark": cfg.benchmark,
        "mode": mode,
        "provider": cfg.data_provider,
        "git_commit": git_commit_hash(),
        "active_modules": dict(zip(modules["module"], modules["active"], strict=False)),
        "feature_counts": {
            "total_features_created": total_features,
            "hmm_features_requested": hmm_requested,
            "hmm_features_used": hmm_used,
            "features_dropped": features_dropped,
        },
        "data_rows": {
            "raw_rows": int(raw_rows or 0),
            "rows_after_features": int(feature_rows or backtest_rows or 0),
            "rows_dropped": max(int(raw_rows or 0) - int(feature_rows or 0), 0),
            "first_date": first_date,
            "last_date": last_date,
        },
        "warnings": warnings,
        "outputs": {
            "artifact_count": int(len(status)),
            "active_output_files": active_files,
            "empty_output_files": empty_files,
            "skipped_output_files": skipped_files,
        },
        "config": {
            "hash": config_hash(cfg),
            "portfolio_enabled": cfg.portfolio_enabled,
            "portfolio_mode": cfg.portfolio_mode,
            "feature_groups_enabled": list(cfg.feature_groups_enabled),
        },
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    status = output_status_frame(cfg, output_dir)
    modules = active_modules_frame(cfg, status)
    status.to_csv(output_dir / "output_status.csv", index=False)
    modules.to_csv(output_dir / "active_modules.csv", index=False)
    summary["active_modules"] = dict(zip(modules["module"], modules["active"], strict=False))
    summary["outputs"] = {
        "artifact_count": int(len(status)),
        "active_output_files": status.loc[status["status"].eq("active"), "file"].tolist(),
        "empty_output_files": status.loc[status["status"].eq("empty"), "file"].tolist(),
        "skipped_output_files": status.loc[
            status["status"].str.startswith("skipped"), "file"
        ].tolist(),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary
