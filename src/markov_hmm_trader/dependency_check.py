from __future__ import annotations

import importlib.util

import pandas as pd


DEPENDENCY_GROUPS = {
    "hmmlearn": "Core HMM engine",
    "statsforecast": "Optional statistical forecast benchmarks and GARCH/ARCH models",
    "torch": "Optional neural-network confirmation",
    "statsmodels": "Optional MarkovRegression benchmark",
    "yfinance": "Market data download",
}


def dependency_health_frame() -> pd.DataFrame:
   