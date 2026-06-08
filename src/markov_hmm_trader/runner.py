from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, replace

import pandas as pd

from .benchmark import run_markov_regression_benchmark
from .backtest import walk_forward_backtest
from .config import PipelineConfig
from .data import load_market_data
from .diagnostics import summarize_states
from .features import make_features
from .metrics import make_report
from .plots import save_plots
from .stress import make_stress_tests


def run_single(
    cfg: PipelineConfig,
    output_dir: Path,
    make_plots: bool = True,
    write_diagnostics: bool | None = None,
    write_stress: bool | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    diagnostics_enabled = cfg.enable_diagnostics if write_diagnostics is None else write_diagnostics
    stress_enabled = cfg.enable_stress_tests if write_stress is None else write_stress
    run_cfg = replace(cfg, enable_diagnostics=diagnostics_enabled, enable_stress_tests=stress_enabled)

    print(f"Loading data for {cfg.ticker} vs {cfg.benchmark}...")
    raw = load_market_data(run_cfg)

    print("Building features...")
    features = make_features(raw, run_cfg)

    print("Running walk-forward HMM backtest...")
    result = walk_forward_backtest(features, run_cfg)
    backtest = result.backtest

    print("Creating report...")
    stats = make_report(backtest, run_cfg)

    output_dir.mkdir(exist_ok=True)
    backtest.to_csv(output_dir / "equity_curve.csv")
    stats.to_frame("value").to_csv(output_dir / "stats.csv")

    if diagnostics_enabled:
        print("Saving diagnostics...")
        result.window_diagnostics.to_csv(output_dir / "window_diagnostics.csv", index=False)
        result.seed_diagnostics.to_csv(output_dir / "seed_diagnostics.csv", index=False)
        summarize_states(backtest, result.window_diagnostics).to_csv(output_dir / "state_summary.csv", index=False)

    if stress_enabled:
        print("Saving stress tests...")
        make_stress_tests(backtest, run_cfg).to_csv(output_dir / "stress_tests.csv", index=False)

    if run_cfg.enable_benchmark:
        print("Saving statsmodels benchmark...")
        run_markov_regression_benchmark(features, run_cfg).to_csv(
            output_dir / "markov_regression_benchmark.csv",
            index=False,
        )

    if make_plots:
        print("Saving plots...")
        save_plots(backtest, run_cfg, output_dir)

    return backtest, stats


def run_universe(tickers: list[str], cfg: PipelineConfig, output_dir: Path) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        print(f"\nRunning {ticker}...")
        try:
            local_cfg = PipelineConfig(**{**asdict(cfg), "ticker": ticker})
            _, stats = run_single(
                local_cfg,
                output_dir / ticker,
                make_plots=False,
                write_diagnostics=False,
                write_stress=False,
            )
            rows.append(stats)
        except Exception as err:
            print(f"Failed {ticker}: {err}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("ticker")
