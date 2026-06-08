import argparse
from pathlib import Path

from src.markov_hmm_trader.config import load_config
from src.markov_hmm_trader.runner import run_single, run_universe

CONFIG_PATH = Path("configs/default.json")
OUTPUT_DIR = Path("outputs")

RUN_UNIVERSE = False

UNIVERSE = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Markov/HMM trading research pipeline.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to a JSON config file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for generated reports.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Skip HMM diagnostics CSV outputs.")
    parser.add_argument("--no-stress", action="store_true", help="Skip stress-test CSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    args.output_dir.mkdir(exist_ok=True)

    if RUN_UNIVERSE:
        universe_stats = run_universe(UNIVERSE, cfg, args.output_dir)
        print("\nUniverse Results:")
        print(universe_stats.sort_values("strategy_sharpe", ascending=False).to_string())
        universe_stats.to_csv(args.output_dir / "universe_stats.csv")
    else:
        backtest, stats = run_single(
            cfg,
            args.output_dir,
            make_plots=not args.no_plots,
            write_diagnostics=False if args.no_diagnostics else None,
            write_stress=False if args.no_stress else None,
        )
        print("\nSingle Ticker Results:")
        print(stats.to_string())


if __name__ == "__main__":
    main()
