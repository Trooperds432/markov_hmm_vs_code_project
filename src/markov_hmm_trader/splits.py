from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SPLIT_REGISTRY_COLUMNS = [
    "split_id",
    "window_id",
    "symbol",
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
    "n_train",
    "n_validation",
    "n_test",
    "purge_bars",
    "embargo_bars",
    "split_method",
    "created_at_utc",
]


def _date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).date().isoformat()


def build_split_registry(
    index: pd.DatetimeIndex,
    symbol: str,
    train_window: int,
    test_window: int,
    validation_fraction: float = 0.2,
    purge_bars: int = 0,
    embargo_bars: int = 0,
    step_size: int | None = None,
    created_at_utc: str | None = None,
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(index).sort_values()
    step = max(int(step_size or test_window), 1)
    train_window = max(int(train_window), 2)
    test_window = max(int(test_window), 1)
    purge_bars = max(int(purge_bars), 0)
    embargo_bars = max(int(embargo_bars), 0)
    created_at_utc = created_at_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    validation_rows = max(1, int(round(train_window * float(validation_fraction))))
    validation_rows = min(validation_rows, train_window - 1)
    rows = []

    window_id = 1
    for train_end_pos in range(train_window, len(dates) - 1, step):
        test_start_pos = train_end_pos + purge_bars + embargo_bars
        test_end_pos = min(test_start_pos + test_window - 1, len(dates) - 1)
        if test_start_pos >= len(dates) or test_end_pos < test_start_pos:
            break

        train_start_pos = train_end_pos - train_window
        validation_start_pos = train_end_pos - validation_rows
        rows.append(
            {
                "split_id": f"{symbol}-{window_id:04d}",
                "window_id": window_id,
                "symbol": symbol,
                "train_start": _date(dates[train_start_pos]),
                "train_end": _date(dates[train_end_pos - validation_rows - 1]),
                "validation_start": _date(dates[validation_start_pos]),
                "validation_end": _date(dates[train_end_pos - 1]),
                "test_start": _date(dates[test_start_pos]),
                "test_end": _date(dates[test_end_pos]),
                "n_train": int(train_window - validation_rows),
                "n_validation": int(validation_rows),
                "n_test": int(test_end_pos - test_start_pos + 1),
                "purge_bars": purge_bars,
                "embargo_bars": embargo_bars,
                "split_method": "walk_forward",
                "created_at_utc": created_at_utc,
            }
        )
        window_id += 1

    return pd.DataFrame(rows, columns=SPLIT_REGISTRY_COLUMNS)


def write_split_registry(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=SPLIT_REGISTRY_COLUMNS).to_csv(path, index=False)
