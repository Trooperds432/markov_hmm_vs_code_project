from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_SELECTION_AUDIT_COLUMNS = [
    "window_id",
    "symbol",
    "candidate_id",
    "model_family",
    "hmm_model_type",
    "n_states",
    "n_mix",
    "covariance_type",
    "train_log_likelihood",
    "validation_log_likelihood",
    "aic",
    "bic",
    "score",
    "selected",
    "rejection_reason",
    "warning_flags",
    "created_at_utc",
]


def _warning_flags(row: pd.Series) -> str:
    flags = []
    if "converged" in row and not bool(row.get("converged")):
        flags.append("non_converged")
    if str(row.get("error", "")):
        flags.append("failed_fit")
    delta = row.get("final_likelihood_delta", np.nan)
    if pd.notna(delta) and float(delta) < -1e-6:
        flags.append("unstable_likelihood")
    return "|".join(flags)


def build_model_selection_candidates(
    model_selection: pd.DataFrame,
    symbol: str,
    created_at_utc: str | None = None,
) -> pd.DataFrame:
    created_at_utc = created_at_utc or datetime.now(UTC).replace(microsecond=0).isoformat()
    if model_selection.empty:
        return pd.DataFrame(columns=MODEL_SELECTION_AUDIT_COLUMNS)

    rows = []
    for position, (_, row) in enumerate(model_selection.iterrows(), start=1):
        candidate_id = row.get("candidate_id", position)
        selected = bool(row.get("selected", False))
        error = str(row.get("error", ""))
        score = row.get("selection_score", np.nan)
        if pd.isna(score):
            score = row.get("validation_log_likelihood", row.get("train_log_likelihood", np.nan))
        rejection_reason = ""
        if not selected:
            rejection_reason = error or "lower_selection_score"
        rows.append(
            {
                "window_id": row.get("window_id", ""),
                "symbol": symbol,
                "candidate_id": candidate_id,
                "model_family": "hmm",
                "hmm_model_type": row.get("model_type", row.get("hmm_model_type", "")),
                "n_states": row.get("n_states", ""),
                "n_mix": row.get("n_mix", ""),
                "covariance_type": row.get("covariance_type", ""),
                "train_log_likelihood": row.get("train_log_likelihood", np.nan),
                "validation_log_likelihood": row.get("validation_log_likelihood", np.nan),
                "aic": row.get("validation_aic", row.get("aic", np.nan)),
                "bic": row.get("validation_bic", row.get("bic", np.nan)),
                "score": score,
                "selected": selected,
                "rejection_reason": rejection_reason,
                "warning_flags": _warning_flags(row),
                "created_at_utc": created_at_utc,
            }
        )
    return pd.DataFrame(rows, columns=MODEL_SELECTION_AUDIT_COLUMNS)


def write_model_selection_candidates(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.reindex(columns=MODEL_SELECTION_AUDIT_COLUMNS).to_csv(path, index=False)
