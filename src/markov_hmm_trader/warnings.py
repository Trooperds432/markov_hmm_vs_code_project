from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

WARNING_COLUMNS = [
    "code",
    "severity",
    "module",
    "message",
    "details",
    "provider",
    "symbol",
    "capability",
    "feature_group",
    "feature",
    "reason",
    "detail",
]


@dataclass(frozen=True)
class StructuredWarning:
    code: str
    severity: str
    module: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def warning_to_row(warning: StructuredWarning, **extra: Any) -> dict[str, Any]:
    row = {
        "code": warning.code,
        "severity": warning.severity,
        "module": warning.module,
        "message": warning.message,
        "details": json.dumps(warning.details, sort_keys=True, default=str),
        "provider": "",
        "symbol": "",
        "capability": "",
        "feature_group": "",
        "feature": "",
        "reason": warning.code,
        "detail": warning.message,
    }
    row.update({key: value for key, value in extra.items() if key in WARNING_COLUMNS})
    if "reason" not in extra:
        row["reason"] = warning.code
    if "detail" not in extra:
        row["detail"] = warning.message
    return row


def warnings_to_frame(warnings: list[StructuredWarning] | list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for warning in warnings:
        if isinstance(warning, StructuredWarning):
            rows.append(warning_to_row(warning))
        else:
            rows.append(warning)
    return normalize_warning_frame(pd.DataFrame(rows))


def append_warning(
    warnings: list[StructuredWarning] | list[dict[str, Any]],
    warning: StructuredWarning,
    **extra: Any,
) -> list[StructuredWarning] | list[dict[str, Any]]:
    warnings.append(warning_to_row(warning, **extra))
    return warnings


def normalize_warning_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=WARNING_COLUMNS)
    out = frame.copy()
    if "reason" not in out and "code" in out:
        out["reason"] = out["code"]
    if "detail" not in out and "message" in out:
        out["detail"] = out["message"]
    if "code" not in out and "reason" in out:
        out["code"] = out["reason"]
    if "message" not in out and "detail" in out:
        out["message"] = out["detail"]
    if "severity" not in out:
        out["severity"] = "warning"
    if "module" not in out:
        out["module"] = ""
    if "details" not in out:
        out["details"] = ""
    for column in WARNING_COLUMNS:
        if column not in out:
            out[column] = ""
    return out[WARNING_COLUMNS]


def warnings_to_json_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    normalized = normalize_warning_frame(frame)
    return normalized.to_dict("records")
