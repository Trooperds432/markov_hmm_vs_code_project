from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_registry(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def latest_blessed_model(path: str | Path, ticker: str) -> pd.Series | None:
    registry = read_registry(path)
    if registry.empty or "approved_for_paper_trading" not in registry:
        return None
    rows = registry[
        (registry.get("ticker") == ticker) & (registry["approved_for_paper_trading"].astype(bool))
    ]
    if rows.empty:
        return None
    return rows.iloc[-1]
