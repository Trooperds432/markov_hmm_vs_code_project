from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import walk_forward_backtest
from .baselines import baseline_comparison, write_baseline_comparison
from .config import PipelineConfig
from .doctor import write_doctor_reports
from .features import all_feature_warnings, feature_manifest, make_features
from .metrics import add_backtest_metric_columns, make_report
from .model_audit import build_model_selection_candidates, write_model_selection_candidates
from .output_contracts import assert_contract_results, validate_output_directory
from .reproducibility import set_global_seed, write_experiment_metadata
from .splits import build_split_registry, write_split_registry


def synthetic_ohlcv(cfg: PipelineConfig, periods: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg.random_seed))
    index = pd.bdate_range("2020-01-01", periods=periods)
    drift = 0.0004 + 0.0002 * np.sin(np.linspace(0, 8 * np.pi, periods))
    noise = rng.normal(0.0, 0.012, periods)
    returns = drift + noise
    benchmark_returns = drift * 0.6 + rng.normal(0.0, 0.008, periods)
    close = 100.0 * np.exp(np.cumsum(returns))
    benchmark_close = 100.0 * np.exp(np.cumsum(benchmark_returns))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, periods))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.002, 0.003, periods)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.002, 0.003, periods)))
    volume = rng.integers(1_000_000, 5_000_000, periods)
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Benchmark_Close": benchmark_close,
        },
        index=index,
    )


def ci_smoke_config(cfg: PipelineConfig) -> PipelineConfig:
    return replace(
        cfg,
        ticker="SYNTH",
        benchmark="SYNB",
        data_provider="synthetic",
        synthetic_periods=360,
        random_seed=int(cfg.random_seed),
        deterministic=True,
        train_window=80,
        retrain_frequency=40,
        n_iter=20,
        tolerance=1e-3,
        covariance_type="diag",
        random_seeds=(int(cfg.random_seed),),
        enable_hmm_model_selection=True,
        hmm_model_types=("gaussian",),
        hmm_state_candidates=(3,),
        hmm_covariance_candidates=("diag",),
        hmm_n_mix_candidates=(1,),
        hmm_min_covar_candidates=(1e-3,),
        hmm_transmat_prior_candidates=(1.0,),
        enable_diagnostics=True,
        enable_stress_tests=False,
        enable_benchmark=False,
        enable_forecast_benchmarks=False,
        enable_volatility_forecasts=False,
        enable_observable_markov=False,
        enable_nn_confirmation=False,
        enable_experiment_tracking=False,
        enable_model_registry=False,
        use_extra_factor_features=False,
    )


def run_ci_smoke(
    cfg: PipelineConfig,
    output_dir: Path = Path("outputs"),
    config_path: Path = Path("configs/default.json"),
) -> dict[str, pd.DataFrame | pd.Series]:
    smoke_cfg = ci_smoke_config(cfg)
    set_global_seed(smoke_cfg.random_seed, smoke_cfg.deterministic)
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    doctor_report = write_doctor_reports(config_path=config_path, output_dir=output_dir)
    if doctor_report["status"] == "fail":
        raise RuntimeError("doctor_failed_required_checks")

    raw = synthetic_ohlcv(smoke_cfg)
    features = make_features(raw, smoke_cfg)
    result = walk_forward_backtest(features, smoke_cfg)
    backtest = add_backtest_metric_columns(result.backtest, smoke_cfg)
    stats = make_report(backtest, smoke_cfg)

    backtest.to_csv(output_dir / "equity_curve.csv")
    stats.rename_axis("metric").to_frame("value").reset_index().to_csv(
        output_dir / "stats.csv", index=False
    )
    feature_manifest(features, smoke_cfg).to_csv(output_dir / "feature_manifest.csv", index=False)
    all_feature_warnings(features).to_csv(output_dir / "warnings.csv", index=False)

    split_registry = build_split_registry(
        features.index,
        smoke_cfg.ticker,
        smoke_cfg.train_window,
        smoke_cfg.retrain_frequency,
        smoke_cfg.hmm_validation_fraction,
        smoke_cfg.purge_bars,
        smoke_cfg.embargo_bars,
    )
    model_candidates = build_model_selection_candidates(
        result.hmm_model_selection,
        smoke_cfg.ticker,
    )
    baselines = baseline_comparison(backtest, smoke_cfg, smoke_cfg.ticker)

    write_split_registry(split_registry, diagnostics_dir / "split_registry.csv")
    write_model_selection_candidates(
        model_candidates,
        diagnostics_dir / "model_selection_candidates.csv",
    )
    write_baseline_comparison(baselines, diagnostics_dir / "baseline_comparison.csv")
    write_experiment_metadata(diagnostics_dir / "experiment_metadata.json", smoke_cfg)

    contract_results = validate_output_directory(output_dir, mode="ci_smoke")
    assert_contract_results(contract_results)
    return {
        "backtest": backtest,
        "stats": stats,
        "split_registry": split_registry,
        "model_selection_candidates": model_candidates,
        "baseline_comparison": baselines,
    }


def run_diagnostics_mode(
    cfg: PipelineConfig,
    output_dir: Path = Path("outputs"),
    config_path: Path = Path("configs/default.json"),
) -> list[object]:
    set_global_seed(cfg.random_seed, cfg.deterministic)
    report = write_doctor_reports(config_path=config_path, output_dir=output_dir)
    results = validate_output_directory(output_dir, mode="diagnostics")
    if report["status"] == "fail":
        raise RuntimeError("doctor_failed_required_checks")
    return results
