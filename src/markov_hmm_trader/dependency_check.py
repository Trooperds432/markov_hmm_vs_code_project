from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DependencySpec:
    package: str
    import_name: str
    required_for: str
    optional: bool


DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("hmmlearn", "hmmlearn", "core HMM engine", False),
    DependencySpec("statsforecast", "statsforecast", "optional forecast benchmarks", True),
    Dependency