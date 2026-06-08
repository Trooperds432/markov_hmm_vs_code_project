from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import PipelineConfig
from .states import STATE_BEAR, STATE_BULL, STATE_SIDE


def save_plots(df: pd.DataFrame, cfg: PipelineConfig, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(14, 6))
    df[["strategy_equity", "buy_hold_equity"]].plot(ax=plt.gca(), linewidth=1.5)
    plt.title(f"{cfg.ticker}: Strategy vs Buy & Hold")
    plt.ylabel("Equity")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "equity_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(14, 6))
    df[[STATE_BULL, STATE_BEAR, STATE_SIDE]].plot(ax=plt.gca(), linewidth=1.0)
    plt.title("HMM Next-State Probabilities")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "regime_probabilities.png", dpi=150)
    plt.close()

    plt.figure(figsize=(14, 4))
    df["position"].plot(ax=plt.gca(), linewidth=1.0)
    plt.title("Target Position Exposure")
    plt.ylabel("Exposure")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "position_exposure.png", dpi=150)
    plt.close()
