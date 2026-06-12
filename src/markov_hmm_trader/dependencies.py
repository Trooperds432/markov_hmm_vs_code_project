from __future__ import annotations

import importlib.util

import pandas as pd

DEPENDENCY_GROUPS = {
    "required": {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "hmmlearn": "hmmlearn",
        "statsmodels": "statsmodels",
    },
    "base_data": {
        "yfinance": "yfinance",
    },
    "forecast": {
        "statsforecast": "statsforecast",
        "arch": "arch",
    },
    "ml": {
        "torch": "torch",
    },
    "openbb_data": {
        "openbb": "openbb",
        "pyarrow": "pyarrow",
    },
    "dev": {
        "pytest": "pytest",
        "ruff": "ruff",
        "black": "black",
    },
}
DEPENDENCIES = {
    dependency: module
    for modules in DEPENDENCY_GROUPS.values()
    for dependency, module in modules.items()
}

OPTIONAL_DEPENDENCIES = {
    *DEPENDENCY_GROUPS["forecast"],
    *DEPENDENCY_GROUPS["ml"],
    *DEPENDENCY_GROUPS["openbb_data"],
}


def dependency_status() -> pd.DataFrame:
    rows = []
    for group, modules in DEPENDENCY_GROUPS.items():
        for name, module in modules.items():
            available = importlib.util.find_spec(module) is not None
            rows.append(
                {
                    "dependency": name,
                    "module": module,
                    "group": group,
                    "required": name not in OPTIONAL_DEPENDENCIES,
                    "status": "available" if available else "missing",
                }
            )
    return pd.DataFrame(rows)


def dependency_status_text() -> str:
    status = dependency_status()
    lines = []
    for group, group_frame in status.groupby("group", sort=False):
        lines.append(f"[{group}]")
        lines.extend(f"{row.dependency}: {row.status}" for row in group_frame.itertuples())
    return "\n".join(lines)
