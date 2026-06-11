from __future__ import annotations

import pandas as pd

from .dependencies import DEPENDENCIES, dependency_status


def check_dependencies() -> pd.DataFrame:
    """Return optional dependency availability in a stable, testable table."""
    return dependency_status().rename(columns={"dependency": "package"})


def format_dependency_report(status: pd.DataFrame | None = None) -> str:
    """Format dependency health for `python main.py --check-deps`."""
    frame = check_dependencies() if status is None else status.copy()
    name_column = "package" if "package" in frame else "dependency"
    return "\n".join(f"{row[name_column]}: {row['status']}" for _, row in frame.iterrows())


def dependency_status_text() -> str:
    return format_dependency_report(check_dependencies())


__all__ = [
    "DEPENDENCIES",
    "check_dependencies",
    "dependency_status_text",
    "format_dependency_report",
]
