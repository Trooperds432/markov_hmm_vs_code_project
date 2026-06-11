import argparse
from pathlib import Path

from src.markov_hmm_trader.config import (
    PROFILE_OVERRIDES,
    apply_overrides,
    apply_profile,
    apply_quick_mode,
    load_config,
)
from src.markov_hmm_trader.config_validation import assert_config_valid, validate_config
from src.markov_hmm_trader.dependency_check import dependency_status_text
from src.markov_hmm_trader.reproducibility import set_global_seed

CONFIG_PATH = Path("configs/default.json")
OUTPUT_DIR = Path("outputs")
RUNTIME_MODES = ("research", "backtest", "paper", "diagnostics", "ci_smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Markov/HMM trading research pipeline.")
    parser.add_argument(
        "--config", type=Path, default=CONFIG_PATH, help="Path to a JSON config file."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for generated reports."
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    plot_group.add_argument("--plots", action="store_true", help="Force PNG plot generation.")
    parser.add_argument(
        "--no-diagnostics", action="store_true", help="Skip HMM diagnostics CSV outputs."
    )
    parser.add_argument("--no-stress", action="store_true", help="Skip stress-test CSV outputs.")
    parser.add_argument("--universe", action="store_true", help="Run the configured universe list.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_OVERRIDES),
        help="Optimisation/risk profile override.",
    )
    parser.add_argument("--ticker", help="Override the configured ticker.")
    parser.add_argument("--benchmark", help="Override the configured benchmark.")
    parser.add_argument("--start", help="Override the configured start date.")
    parser.add_argument("--end", help="Override the configured end date.")
    parser.add_argument(
        "--quick", action="store_true", help="Use a lighter debugging configuration."
    )
    parser.add_argument(
        "--nested", action="store_true", help="Run nested walk-forward optimisation."
    )
    parser.add_argument("--portfolio", action="store_true", help="Run portfolio construction mode.")
    parser.add_argument(
        "--compare-models",
        "--model-comparison",
        dest="compare_models",
        action="store_true",
        help="Run model comparison mode.",
    )
    parser.add_argument(
        "--paper", action="store_true", help="Run simulation-only paper trading mode."
    )
    parser.add_argument(
        "--mode",
        choices=RUNTIME_MODES,
        help="Explicit runtime mode. Existing mode flags remain supported.",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run environment and output-directory health diagnostics.",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Print optional dependency availability and exit.",
    )
    return parser.parse_args()


def resolve_runtime_mode(args: argparse.Namespace) -> str:
    legacy_modes = {
        "universe": args.universe,
        "nested": args.nested,
        "portfolio": args.portfolio,
        "compare_models": args.compare_models,
        "paper": args.paper,
    }
    selected_legacy = [name for name, enabled in legacy_modes.items() if enabled]
    if len(selected_legacy) > 1:
        names = ", ".join(f"--{name.replace('_', '-')}" for name in selected_legacy)
        raise SystemExit(f"Incompatible runtime modes supplied: {names}. Choose only one.")

    if args.mode is None:
        return selected_legacy[0] if selected_legacy else "research"

    if selected_legacy:
        legacy = selected_legacy[0]
        if not (args.mode == "paper" and legacy == "paper"):
            raise SystemExit(
                "Incompatible runtime modes supplied: "
                f"--mode {args.mode} cannot be combined with --{legacy.replace('_', '-')}."
            )
    return args.mode


def main() -> None:
    args = parse_args()
    if args.check_deps:
        print(dependency_status_text())
        return
    if args.doctor:
        from src.markov_hmm_trader.doctor import format_dependency_report_text, write_doctor_reports

        report = write_doctor_reports(config_path=args.config, output_dir=args.output_dir)
        print(format_dependency_report_text(report))
        if report["status"] == "fail":
            raise SystemExit(1)
        return

    from src.markov_hmm_trader.runner import (
        run_model_comparison_mode,
        run_nested_mode,
        run_paper_mode,
        run_portfolio_mode,
        run_single,
        run_universe,
    )
    from src.markov_hmm_trader.runtime_modes import run_ci_smoke, run_diagnostics_mode

    cfg = load_config(args.config)
    cfg = apply_profile(cfg, args.profile)
    cfg = apply_overrides(
        cfg,
        ticker=args.ticker,
        benchmark=args.benchmark,
        start=args.start,
        end=args.end,
    )
    if args.quick:
        cfg = apply_quick_mode(cfg)
    for issue in validate_config(cfg):
        if issue.severity == "warning":
            print(f"Config warning: {issue.field}: {issue.message}")
    assert_config_valid(cfg)
    set_global_seed(cfg.random_seed, cfg.deterministic)

    mode = resolve_runtime_mode(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_plots = args.plots or (not args.no_plots and not args.quick)

    if mode == "ci_smoke":
        outputs = run_ci_smoke(cfg, args.output_dir, config_path=args.config)
        print("\nCI Smoke Results:")
        print(outputs["stats"].to_string())
    elif mode == "diagnostics":
        results = run_diagnostics_mode(cfg, args.output_dir, config_path=args.config)
        print("\nDiagnostics contract checks:")
        for result in results:
            print(f"{result.name}: {result.status} ({result.message})")
    elif mode == "universe":
        universe_stats = run_universe(list(cfg.universe), cfg, args.output_dir)
        print("\nUniverse Results:")
        print(universe_stats.sort_values("strategy_sharpe", ascending=False).to_string())
        universe_stats.to_csv(args.output_dir / "universe_stats.csv")
    elif mode == "nested":
        run_nested_mode(cfg, args.output_dir, make_plots=make_plots)
    elif mode == "portfolio":
        run_portfolio_mode(cfg, args.output_dir, make_plots=make_plots)
    elif mode == "compare_models":
        run_model_comparison_mode(cfg, args.output_dir, make_plots=make_plots)
    elif mode == "paper":
        run_paper_mode(cfg, args.output_dir, make_plots=make_plots)
    else:
        backtest, stats = run_single(
            cfg,
            args.output_dir,
            make_plots=make_plots,
            write_diagnostics=False if args.no_diagnostics else None,
            write_stress=False if args.no_stress else None,
        )
        print("\nSingle Ticker Results:")
        print(stats.to_string())


if __name__ == "__main__":
    main()
