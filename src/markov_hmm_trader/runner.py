from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .backtest import walk_forward_backtest
from .benchmark import run_markov_regression_benchmark
from .config import PipelineConfig
from .data import factor_data_coverage, load_market_data
from .diagnostics import summarize_states
from .ensemble import ensemble_diagnostics
from .experiment import finalize_experiment_run, start_experiment_run
from .features import feature_summary, make_features
from .metrics import add_backtest_metric_columns, make_report
from .nn_model import nn_predictions_frame
from .plots import save_plots
from .stress import (
    make_asset_bootstrap_stress,
    make_cost_slippage_stress,
    make_crisis_window_stress,
    make_missing_data_stress,
    make_stress_tests,
)


def run_single(
    cfg: PipelineConfig,
    output_dir: Path,
    make_plots: bool = True,
    write_diagnostics: bool | None = None,
    write_stress: bool | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    diagnostics_enabled = cfg.enable_diagnostics if write_diagnostics is None else write_diagnostics
    stress_enabled = cfg.enable_stress_tests if write_stress is None else write_stress
    run_cfg = replace(
        cfg, enable_diagnostics=diagnostics_enabled, enable_stress_tests=stress_enabled
    )
    experiment_dir = start_experiment_run(run_cfg, output_dir)

    print(f"Loading data for {cfg.ticker} vs {cfg.benchmark}...")
    raw = load_market_data(run_cfg)

    print("Building features...")
    features = make_features(raw, run_cfg)

    print("Running walk-forward HMM backtest...")
    result = walk_forward_backtest(features, run_cfg)
    backtest = add_backtest_metric_columns(result.backtest, run_cfg)

    print("Creating report...")
    stats = make_report(backtest, run_cfg)

    output_dir.mkdir(exist_ok=True)
    backtest.to_csv(output_dir / "equity_curve.csv")
    stats.to_frame("value").to_csv(output_dir / "stats.csv")

    if diagnostics_enabled:
        print("Saving diagnostics...")
        factor_data_coverage(raw).to_csv(output_dir / "factor_data_coverage.csv", index=False)
        feature_summary(features, run_cfg).to_csv(output_dir / "feature_summary.csv", index=False)
        result.window_diagnostics.to_csv(output_dir / "window_diagnostics.csv", index=False)
        result.seed_diagnostics.to_csv(output_dir / "seed_diagnostics.csv", index=False)
        summarize_states(backtest, result.window_diagnostics).to_csv(
            output_dir / "state_summary.csv", index=False
        )
        result.hmm_model_selection.to_csv(output_dir / "hmm_model_selection.csv", index=False)
        result.hmm_emissions.to_csv(output_dir / "hmm_emissions.csv", index=False)
        result.hmm_transition_matrices.to_csv(
            output_dir / "hmm_transition_matrices.csv",
            index=False,
        )
        result.hmm_state_stability.to_csv(output_dir / "hmm_state_stability.csv", index=False)
        nn_predictions_frame(backtest).to_csv(output_dir / "nn_predictions.csv")
        ensemble_diagnostics(backtest).to_csv(output_dir / "ensemble_diagnostics.csv")

    if stress_enabled:
        print("Saving stress tests...")
        make_stress_tests(backtest, run_cfg).to_csv(output_dir / "stress_tests.csv", index=False)
        make_asset_bootstrap_stress(backtest, run_cfg).to_csv(
            output_dir / "asset_bootstrap_stress.csv",
            index=False,
        )
        make_cost_slippage_stress(backtest, run_cfg).to_csv(
            output_dir / "cost_slippage_stress.csv",
            index=False,
        )
        make_missing_data_stress(backtest, run_cfg).to_csv(
            output_dir / "missing_data_stress.csv",
            index=False,
        )
        make_crisis_window_stress(backtest, run_cfg).to_csv(
            output_dir / "crisis_window_stress.csv",
            index=False,
        )

    if run_cfg.enable_benchmark:
        print("Saving statsmodels benchmark...")
        run_markov_regression_benchmark(features, run_cfg).to_csv(
            output_dir / "markov_regression_benchmark.csv",
            index=False,
        )

    if make_plots:
        print("Saving plots...")
        save_plots(backtest, run_cfg, output_dir)

    selected_model = {}
    if not result.hmm_transition_matrices.empty:
        latest_model = result.hmm_transition_matrices.iloc[-1]
        selected_model = {
            "selected_n_states": latest_model.get("selected_n_states", run_cfg.n_states),
            "covariance_type": latest_model.get(
                "selected_covariance_type", run_cfg.covariance_type
            ),
        }
    finalize_experiment_run(run_cfg, experiment_dir, output_dir, stats, selected_model)
    return backtest, stats


def run_universe(tickers: list[str], cfg: PipelineConfig, output_dir: Path) -> pd.DataFrame:
    rows = []

    for ticker in tickers:
        print(f"\nRunning {ticker}...")
        try:
            local_cfg = PipelineConfig(
                **{**asdict(cfg), "ticker": ticker, "enable_experiment_tracking": False}
            )
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


def run_nested_mode(cfg: PipelineConfig, output_dir: Path, make_plots: bool = False) -> None:
    from .nested_optimiser import run_nested_optimisation

    print("Running nested walk-forward optimisation...")
    outputs = run_nested_optimisation(replace(cfg, enable_nested_optimisation=True), output_dir)
    print(outputs["nested_window_results"].to_string(index=False))


def run_portfolio_mode(cfg: PipelineConfig, output_dir: Path, make_plots: bool = False) -> None:
    from .portfolio import run_portfolio

    print("Running portfolio construction mode...")
    outputs = run_portfolio(replace(cfg, portfolio_mode=True), output_dir)
    stats = outputs["portfolio_stats"]
    if isinstance(stats, pd.Series):
        print(stats.to_string())


def run_model_comparison_mode(
    cfg: PipelineConfig, output_dir: Path, make_plots: bool = False
) -> None:
    from .model_comparison import run_model_comparison

    print("Loading data for model comparison...")
    raw = load_market_data(cfg)
    features = make_features(raw, cfg)
    outputs = run_model_comparison(features, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=name != "model_comparison_equity_curves")
    if make_plots and not outputs["model_comparison_equity_curves"].empty:
        import matplotlib.pyplot as plt

        outputs["model_comparison_equity_curves"].plot(figsize=(14, 6))
        plt.title("Model Comparison Equity Curves")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_dir / "model_comparison_equity.png", dpi=150)
        plt.close()
    print(outputs["model_comparison"].to_string(index=False))


def run_paper_mode(cfg: PipelineConfig, output_dir: Path, make_plots: bool = False) -> None:
    from .execution.order_manager import orders_frame, target_weight_order
    from .execution.paper_broker import PaperBroker
    from .execution.risk_checks import run_risk_checks

    print("Running simulation-only paper trading mode...")
    paper_cfg = replace(cfg, enable_paper_trading=True, enable_stress_tests=False)
    backtest, _ = run_single(paper_cfg, output_dir, make_plots=make_plots, write_stress=False)
    latest = backtest.iloc[-1]
    target_weight = float(latest.get("position", 0.0))
    broker = PaperBroker(
        initial_cash=paper_cfg.paper_initial_cash,
        slippage_bps=paper_cfg.slippage_bps,
    )
    checks = run_risk_checks(paper_cfg, latest, target_weight, broker.get_cash())
    output_dir.mkdir(parents=True, exist_ok=True)
    checks.to_csv(output_dir / "paper_risk_checks.csv", index=False)

    orders = []
    if checks["passed"].all():
        orders.append(
            target_weight_order(
                broker,
                paper_cfg.ticker,
                target_weight,
                float(latest["Close"]),
                broker.get_cash(),
            )
        )
    else:
        print("Paper trading blocked by risk checks or unapproved model registry entry.")

    orders_frame(orders).to_csv(output_dir / "paper_orders.csv", index=False)
    pd.DataFrame([broker.get_positions()]).to_csv(output_dir / "paper_positions.csv", index=False)
    pd.DataFrame(
        [
            {
                "cash": broker.get_cash(),
                "equity": broker.equity({paper_cfg.ticker: float(latest["Close"])}),
            }
        ]
    ).to_csv(output_dir / "paper_equity.csv", index=False)
