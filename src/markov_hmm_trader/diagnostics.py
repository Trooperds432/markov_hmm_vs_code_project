from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .hmm_model import HMMFitResult
from .states import STATE_ORDER


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def expected_dwell_times(transmat: np.ndarray) -> dict[int, float]:
    dwell = {}
    for component, self_prob in enumerate(np.diag(transmat)):
        if self_prob >= 1.0:
            dwell[component] = np.inf
        else:
            dwell[component] = float(1.0 / max(1.0 - self_prob, 1e-12))
    return dwell


def posterior_entropy(probs: pd.DataFrame) -> float:
    values = probs[STATE_ORDER].clip(lower=1e-12).to_numpy(dtype=float)
    entropy = -(values * np.log(values)).sum(axis=1)
    return _safe_float(np.mean(entropy))


def make_window_diagnostic_row(
    window_id: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    fit: HMMFitResult,
    next_probs: pd.DataFrame,
) -> dict[str, Any]:
    model = fit.model
    transmat = np.asarray(model.transmat_, dtype=float)
    dwell = expected_dwell_times(transmat)
    semantic_components = {
        state: [component for component, mapped_state in fit.state_map.items() if mapped_state == state]
        for state in STATE_ORDER
    }

    row: dict[str, Any] = {
        "window_id": window_id,
        "train_start": train.index[0].date().isoformat(),
        "train_end": train.index[-1].date().isoformat(),
        "test_start": test.index[0].date().isoformat(),
        "test_end": test.index[-1].date().isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "best_seed": int(fit.best_seed),
        "train_log_likelihood": _safe_float(fit.train_log_likelihood),
        "train_log_likelihood_per_row": _safe_float(fit.train_log_likelihood / max(len(train), 1)),
        "best_converged": bool(getattr(model.monitor_, "converged", False)),
        "best_iterations": int(getattr(model.monitor_, "iter", 0)),
        "posterior_entropy_mean": posterior_entropy(next_probs),
        "state_map": json.dumps({int(k): v for k, v in fit.state_map.items()}, sort_keys=True),
        "transition_matrix": json.dumps(transmat.round(8).tolist()),
    }

    for state, components in semantic_components.items():
        occupancy = sum(fit.component_occupancy.get(component, 0.0) for component in components)
        weighted_returns = [
            fit.component_forward_returns.get(component, np.nan) * fit.component_occupancy.get(component, 0.0)
            for component in components
            if np.isfinite(fit.component_forward_returns.get(component, np.nan))
        ]
        return_denom = sum(
            fit.component_occupancy.get(component, 0.0)
            for component in components
            if np.isfinite(fit.component_forward_returns.get(component, np.nan))
        )
        dwell_values = [dwell[component] for component in components if component in dwell]

        row[f"{state}_components"] = "|".join(str(component) for component in components)
        row[f"{state}_occupancy"] = _safe_float(occupancy)
        row[f"{state}_forward_return"] = _safe_float(sum(weighted_returns) / return_denom) if return_denom else np.nan
        row[f"{state}_expected_dwell"] = _safe_float(np.mean(dwell_values)) if dwell_values else np.nan

    return row


def make_seed_diagnostic_rows(window_id: int, train_end: pd.Timestamp, fit: HMMFitResult) -> list[dict[str, Any]]:
    rows = []
    for item in fit.seed_diagnostics:
        rows.append(
            {
                "window_id": window_id,
                "train_end": train_end.date().isoformat(),
                "seed": int(item.seed),
                "score": _safe_float(item.score),
                "converged": bool(item.converged),
                "iterations": int(item.iterations),
                "error": item.error,
                "is_best": int(item.seed == fit.best_seed),
            }
        )
    return rows


def summarize_states(backtest: pd.DataFrame, window_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in STATE_ORDER:
        decoded_mask = backtest["hidden_state"] == state
        prob = backtest[state] if state in backtest else pd.Series(dtype=float)

        row = {
            "state": state,
            "decoded_rows": int(decoded_mask.sum()),
            "decoded_occupancy": _safe_float(decoded_mask.mean()),
            "avg_probability": _safe_float(prob.mean()),
            "avg_asset_return_when_decoded": _safe_float(backtest.loc[decoded_mask, "ret_1"].mean()),
            "avg_strategy_return_when_decoded": _safe_float(backtest.loc[decoded_mask, "strategy_ret"].mean()),
        }

        if not window_diagnostics.empty:
            dwell_col = f"{state}_expected_dwell"
            fwd_col = f"{state}_forward_return"
            occ_col = f"{state}_occupancy"
            row["avg_train_expected_dwell"] = _safe_float(window_diagnostics.get(dwell_col, pd.Series(dtype=float)).mean())
            row["avg_train_forward_return"] = _safe_float(window_diagnostics.get(fwd_col, pd.Series(dtype=float)).mean())
            row["avg_train_occupancy"] = _safe_float(window_diagnostics.get(occ_col, pd.Series(dtype=float)).mean())
        else:
            row["avg_train_expected_dwell"] = np.nan
            row["avg_train_forward_return"] = np.nan
            row["avg_train_occupancy"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)
