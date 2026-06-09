from __future__ import annotations

import importlib.util

import pandas as pd


DEPENDENCIES = {
    "hmmlearn": "hmmlearn",
    "statsforecast": "statsforecast",
    "torch": "torch",
    "statsmodels": "statsmodels",
    "yfinance": "yfinance",
}


def dependency_status() -> pd.DataFrame:
    rows = []
    for name, module in DEPENDENCIES.items():
        available = importlib.util.find_spec(module) is not None
        rows.append(
            {
                "dependency": name,
                "module": module,
                "status": "available" if available else "missing",
            }
        )
    return pd.DataFrame(rows)


def dependency_status_text() -> str:
    status = dependency_status()
    return "\n".join(f"{row.dependency}: {row.status}" for row in status.itertuples())
