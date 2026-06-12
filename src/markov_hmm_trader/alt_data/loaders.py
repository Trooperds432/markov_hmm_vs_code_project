from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..warnings import StructuredWarning, warning_to_row
from .schemas import GLASSDOOR_ALIASES, VALUATION_ALIASES, WSB_ALIASES, canonicalize_columns


def _warning(code: str, detail: str) -> pd.DataFrame:
    domain = code.split("_", maxsplit=1)[0]
    return pd.DataFrame(
        [
            warning_to_row(
                StructuredWarning(
                    code=code,
                    severity="warning",
                    module="alt_data.loaders",
                    message=f"Alternative data warning: {code}.",
                    details={"detail": detail},
                ),
                feature_group=domain,
                reason=code,
                detail=detail,
            )
        ]
    )


def read_local_table(path: str, missing_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path:
        return pd.DataFrame(), _warning(missing_code, "empty_path")
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame(), _warning(missing_code, str(table_path))
    try:
        if table_path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(table_path), pd.DataFrame()
        return pd.read_csv(table_path), pd.DataFrame()
    except Exception as err:
        return pd.DataFrame(), _warning(missing_code, f"{type(err).__name__}: {err}")


def _normalize(raw: pd.DataFrame, aliases: dict[str, str], date_col: str) -> pd.DataFrame:
    if raw.empty:
        return raw
    out = raw.rename(columns=canonicalize_columns(list(raw.columns), aliases)).copy()
    if date_col in out:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.tz_localize(None)
    if "ticker" in out:
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out


def load_glassdoor_reviews(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, warnings = read_local_table(path, "glassdoor_file_missing")
    return _normalize(raw, GLASSDOOR_ALIASES, "review_date"), warnings


def load_wsb_mentions(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, warnings = read_local_table(path, "wsb_file_missing")
    return _normalize(raw, WSB_ALIASES, "date"), warnings


def load_valuation_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw, warnings = read_local_table(path, "valuation_file_missing")
    return _normalize(raw, VALUATION_ALIASES, "date"), warnings
