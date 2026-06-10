from __future__ import annotations

import importlib.util

import pandas as pd

DEPENDENCY_GROUPS = {
    "hmmlearn": ("hmmlearn", "core HMM engine", False),
    "statsforecast": ("statsforecast", "optional forecast benchmarks", True),
    "torch": ("torch", "optional neural-network confirmation", True),
    "statsmodels": ("statsmodels", "optional Mark