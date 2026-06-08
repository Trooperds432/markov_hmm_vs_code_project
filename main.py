import argparse
from pathlib import Path

from src.markov_hmm_trader.config import (
    PROFILE_OVERRIDES,
    apply_overrides,
    apply_profile,
    apply_quick_mode,
    load_config,
)
from src.markov_hmm_trader.runner import (
    run_model_comparison_mode,
    run_nested_mode,
    run_paper_mode,
    run_portfolio_mode,
    run_single,
    run_universe,
)

CONFIG_PATH = Path("configs/default.json")
OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Markov/HMM trading research pipeline.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to a JSON config file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for generated reports.")
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    plot_group.add_argument("--plots", action="store_true", help="Force PNG plot generation.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip HMM diagnostics CSV outputs.")
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
    parser.add_argument("--quick", action="store_true", help="Use a lighter debugging configuration.")
    parser.add_argument("--nested", action="store_true", help="Run nested walk-forward optimisation.")
    parser.add_argument("--portfolio", action="store_true", help="Run portfolio construction mode.")
    parser.add_argument("--compare-models", action="store_true", help="Run model comparison mode.")
    parser.add_argument("--paper", action="store_true", help="Run simulation-only paper trading mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    mode_flags = [args.universe, args.nested, args.portfolio, args.compare_models, args.paper]
    if sum(bool(flag) for flag in mode_flags) > 1:
        raise SystemExit("Choose only one run mode: --universe, --nested, --portfolio, --compare-models, or --paper.")

    args.output_dir.mkdir(exist_ok=True)
    make_plots = args.plots or (not args.no_plots and not args.quick)

    if args.universe:
        universe_stats = run_universe(list(cfg.universe), cfg, args.output_dir)
        print("\nUniverse Results:")
        print(universe_stats.sort_values("strategy_sharpe", ascending=False).to_string())
        universe_stats.to_csv(args.output_dir / "universe_stats.csv")
    elif args.nested:
        run_nested_mode(cfg, args.output_dir, make_plots=make_plots)
    elif args.portfolio:
        run_portfolio_mode(cfg, args.output_dir, make_plots=make_plots)
    elif args.compare_models:
        run_model_comparison_mode(cfg, args.output_dir, make_plots=make_plots)
    elif args.paper:
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
