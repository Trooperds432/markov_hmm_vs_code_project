from __future__ import annotations

import importlib.util

DEPENDENCIES = ("hmmlearn", "statsforecast", "torch", "statsmodels", "yfinance")


def dependency_health_frame():
    import pandas as pd

    return pd.DataFrame(
        [
            {"package": name, "installed": importlib.util.find_spec(name) is not None}
            for name in DEPENDENCIES
        ]
    )


def print_dependency_health() -> None:
    print(dependency_health_frame().to_string(index=False))
