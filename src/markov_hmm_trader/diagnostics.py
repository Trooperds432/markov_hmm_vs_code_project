from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .hmm_model import HMMFitResult, component_covariance, component_mean, semantic_probabilities
from .states import STATE_BEAR, STATE_BULL, STATE_ORDER, STATE_SIDE


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


def stationary_distribution(fit: HMMFitResult) -> dict[int, float]:
    try:
        stationary = np.asarray(fit.model.get_stationary_distribution(), dtype=float)
    except Exception:
        transmat = np.asarray(fit.model.transmat_, dtype=float)
        values, vectors = np.linalg.eig(transmat.T)
        index = int(np.argmin(np.abs(values - 1.0)))
        stationary = np.real(vectors[:, index])
        stationary = np.maximum(stationary, 0.0)
        stationary = stationary / stationary.sum() if stationary.sum() else stationary
    return {component: _safe_float(value) for component, value in enumerate(stationary)}


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
        state: [
            component for component, mapped_state in fit.state_map.items() if mapped_state == state
        ]
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
        "selected_model_type": fit.selected_model_type,
        "selected_states": int(fit.selected_n_states),
        "selected_covariance": fit.selected_covariance_type,
        "selected_n_mix": int(fit.selected_n_mix),
        "selection_metric": "",
        "selection_score": np.nan,
        "train_log_likelihood": _safe_float(fit.train_log_likelihood),
        "validation_log_likelihood": np.nan,
        "test_log_likelihood": np.nan,
        "train_aic": np.nan,
        "train_bic": np.nan,
        "decode_disagreement_rate": np.nan,
        "train_log_likelihood_per_row": _safe_float(fit.train_log_likelihood / max(len(train), 1)),
        "best_converged": bool(getattr(model.monitor_, "converged", False)),
        "best_iterations": int(getattr(model.monitor_, "iter", 0)),
        "posterior_entropy_mean": posterior_entropy(next_probs),
        "state_map": json.dumps({int(k): v for k, v in fit.state_map.items()}, sort_keys=True),
        "transition_matrix": json.dumps(transmat.round(8).tolist()),
    }
    selected_rows = [row for row in fit.model_selection_rows if row.get("selected")]
    if selected_rows:
        selected = selected_rows[-1]
        row["selection_metric"] = selected.get("selection_metric", "")
        row["selection_score"] = _safe_float(selected.get("selection_score", np.nan))
        row["validation_log_likelihood"] = _safe_float(
            selected.get("validation_log_likelihood", np.nan)
        )
        row["train_aic"] = _safe_float(selected.get("train_aic", np.nan))
        row["train_bic"] = _safe_float(selected.get("train_bic", np.nan))

    stationary = stationary_distribution(fit)

    for state, components in semantic_components.items():
        occupancy = sum(fit.component_occupancy.get(component, 0.0) for component in components)
        weighted_returns = [
            fit.component_forward_returns.get(component, np.nan)
            * fit.component_occupancy.get(component, 0.0)
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
        row[f"{state}_forward_return"] = (
            _safe_float(sum(weighted_returns) / return_denom) if return_denom else np.nan
        )
        row[f"{state}_expected_dwell"] = (
            _safe_float(np.mean(dwell_values)) if dwell_values else np.nan
        )
        row[f"stationary_{state.lower()}"] = _safe_float(
            sum(stationary.get(component, 0.0) for component in components)
        )

    return row


def make_seed_diagnostic_rows(
    window_id: int, train_end: pd.Timestamp, fit: HMMFitResult
) -> list[dict[str, Any]]:
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


def _state_components(state_map: dict[int, str], state: str) -> list[int]:
    return sorted(component for component, mapped in state_map.items() if mapped == state)


def _covariance_for_component(fit: HMMFitResult, component: int) -> np.ndarray:
    covars = np.asarray(fit.model.covars_, dtype=float)
    if covars.ndim == 3:
        return covars[component]
    if covars.ndim == 2:
        values = covars[component]
        return np.diag(values)
    values = np.asarray(covars[component], dtype=float)
    return np.diag(np.atleast_1d(values))


def make_emission_diagnostic_rows(
    window_id: int,
    train: pd.DataFrame,
    fit: HMMFitResult,
    features: list[str],
) -> list[dict[str, Any]]:
    rows = []
    dwell = expected_dwell_times(np.asarray(fit.model.transmat_, dtype=float))
    for component in range(fit.model.n_components):
        covariance = component_covariance(fit.model, component)
        mean = component_mean(fit.model, component)
        diag = np.diag(covariance)
        row: dict[str, Any] = {
            "window_id": window_id,
            "component": component,
            "semantic_state": fit.state_map.get(component, ""),
            "model_type": fit.selected_model_type,
            "covariance_type": fit.selected_covariance_type,
            "occupancy": _safe_float(fit.component_occupancy.get(component, np.nan)),
            "stationary_probability": _safe_float(stationary_distribution(fit).get(component, np.nan)),
            "forward_return": _safe_float(fit.component_forward_returns.get(component, np.nan)),
            "expected_dwell": _safe_float(dwell.get(component, np.nan)),
            "covariance_trace": _safe_float(np.trace(covariance)),
            "covariance_mean_diag": _safe_float(np.mean(diag)),
            "covariance_json": json.dumps(covariance.round(8).tolist()),
            "train_start": train.index[0].date().isoformat(),
            "train_end": train.index[-1].date().isoformat(),
        }
        for index, feature in enumerate(features):
            row[f"emission_mean_{feature}"] = _safe_float(mean[index])
            row[f"mean__{feature}"] = _safe_float(mean[index])
            row[f"emission_cov_diag_{feature}"] = _safe_float(diag[index])
        rows.append(row)
    return rows


def make_transition_matrix_row(
    window_id: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    fit: HMMFitResult,
) -> dict[str, Any]:
    return {
        "window_id": window_id,
        "train_start": train.index[0].date().isoformat(),
        "train_end": train.index[-1].date().isoformat(),
        "test_start": test.index[0].date().isoformat(),
        "test_end": test.index[-1].date().isoformat(),
        "selected_n_states": int(fit.selected_n_states),
        "model_type": fit.selected_model_type,
        "n_states": int(fit.selected_n_states),
        "selected_covariance_type": fit.selected_covariance_type,
        "covariance_type": fit.selected_covariance_type,
        "transition_matrix_json": json.dumps(np.asarray(fit.model.transmat_).round(8).tolist()),
        "state_map_json": json.dumps({int(k): v for k, v in fit.state_map.items()}, sort_keys=True),
        "expected_dwell_json": json.dumps(expected_dwell_times(np.asarray(fit.model.transmat_))),
        "transition_matrix": json.dumps(np.asarray(fit.model.transmat_).round(8).tolist()),
        "semantic_state_map": json.dumps(
            {int(k): v for k, v in fit.state_map.items()}, sort_keys=True
        ),
    }


def make_stationary_distribution_rows(window_id: int, fit: HMMFitResult) -> list[dict[str, Any]]:
    stationary = stationary_distribution(fit)
    dwell = expected_dwell_times(np.asarray(fit.model.transmat_, dtype=float))
    return [
        {
            "window_id": window_id,
            "component": component,
            "semantic_state": fit.state_map.get(component, ""),
            "stationary_probability": _safe_float(probability),
            "occupancy": _safe_float(fit.component_occupancy.get(component, np.nan)),
            "expected_dwell": _safe_float(dwell.get(component, np.nan)),
        }
        for component, probability in stationary.items()
    ]


def make_component_diagnostic_rows(window_id: int, train: pd.DataFrame, fit: HMMFitResult) -> list[dict[str, Any]]:
    rows = []
    stationary = stationary_distribution(fit)
    dwell = expected_dwell_times(np.asarray(fit.model.transmat_, dtype=float))
    hidden = fit.model.predict(fit.scaler.transform(train[fit.feature_names].values)) if hasattr(fit, "feature_names") else None
    for component, stats in fit.component_diagnostics.items():
        state_returns = train["ret_1"].shift(-1)
        if hidden is not None:
            returns = state_returns[hidden == component].dropna()
        else:
            returns = pd.Series(dtype=float)
        downside = returns[returns < 0.0]
        equity = (1.0 + returns.fillna(0.0)).cumprod()
        max_dd = (equity / equity.cummax() - 1.0).min() if not equity.empty else np.nan
        row = {
            "window_id": window_id,
            "component": int(component),
            "semantic_state": fit.state_map.get(component, ""),
            "occupancy": _safe_float(stats.get("occupancy", np.nan)),
            "stationary_probability": _safe_float(stationary.get(component, np.nan)),
            "mean_forward_return": _safe_float(stats.get("forward_return", np.nan)),
            "median_forward_return": _safe_float(returns.median()),
            "volatility": _safe_float(stats.get("volatility", np.nan)),
            "downside_volatility": _safe_float(downside.std(ddof=0)),
            "max_drawdown_when_state": _safe_float(max_dd),
            "expected_dwell": _safe_float(dwell.get(component, np.nan)),
            "trend_alignment": _safe_float(stats.get("trend_score", np.nan)),
            "bull_score": _safe_float(stats.get("bull_score", stats.get("semantic_score", np.nan))),
            "bear_score": _safe_float(stats.get("bear_score", -stats.get("semantic_score", np.nan))),
            "sideways_score": _safe_float(stats.get("sideways_score", np.nan)),
        }
        rows.append(row)
    return rows


def make_decode_comparison_rows(
    window_id: int,
    dates: pd.Index,
    decoded_viterbi_raw: np.ndarray,
    decoded_map_raw: np.ndarray,
    semantic_probs: pd.DataFrame,
    fit: HMMFitResult,
) -> list[dict[str, Any]]:
    rows = []
    for idx, date in enumerate(dates):
        vit_component = int(decoded_viterbi_raw[idx])
        map_component = int(decoded_map_raw[idx])
        vit_state = fit.state_map.get(vit_component, "")
        map_state = fit.state_map.get(map_component, "")
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "window_id": window_id,
                "decoded_component_viterbi": vit_component,
                "decoded_component_map": map_component,
                "decoded_state_viterbi": vit_state,
                "decoded_state_map": map_state,
                "state_decode_disagreement": bool(vit_state != map_state),
                "bull_probability": _safe_float(semantic_probs.iloc[idx][STATE_BULL]),
                "bear_probability": _safe_float(semantic_probs.iloc[idx][STATE_BEAR]),
                "sideways_probability": _safe_float(semantic_probs.iloc[idx][STATE_SIDE]),
            }
        )
    return rows


def make_synthetic_sample_stress(
    fits: list[tuple[int, HMMFitResult]],
    cfg,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    n_paths = max(int(getattr(cfg, "hmm_synthetic_sample_paths", 0)), 0)
    length = max(int(getattr(cfg, "hmm_synthetic_sample_length", 1)), 1)
    for window_id, fit in fits:
        for path_id in range(1, n_paths + 1):
            try:
                _, states = fit.model.sample(length, random_state=fit.best_seed + path_id)
                semantic = [fit.state_map.get(int(state), STATE_SIDE) for state in states]
                signal = np.array(
                    [1.0 if state == STATE_BULL else -1.0 if state == STATE_BEAR else 0.0 for state in semantic]
                )
                rows.append(
                    {
                        "model_id": window_id,
                        "path_id": path_id,
                        "bull_occupancy": float(np.mean(np.array(semantic) == STATE_BULL)),
                        "bear_occupancy": float(np.mean(np.array(semantic) == STATE_BEAR)),
                        "sideways_occupancy": float(np.mean(np.array(semantic) == STATE_SIDE)),
                        "transition_count": int(np.sum(np.asarray(states[1:]) != np.asarray(states[:-1]))),
                        "turnover_proxy": float(np.mean(np.abs(np.diff(signal)))) if len(signal) > 1 else 0.0,
                        "synthetic_signal_mean": float(signal.mean()),
                        "synthetic_signal_std": float(signal.std(ddof=0)),
                        "synthetic_outcome_proxy": float(signal.mean() / (signal.std(ddof=0) + 1e-12)),
                        "synthetic_drawdown_proxy": float(np.minimum.accumulate(np.cumsum(signal)).min()),
                    }
                )
            except Exception as err:
                rows.append({"model_id": window_id, "path_id": path_id, "error": type(err).__name__})
    stress = pd.DataFrame(rows)
    if stress.empty:
        return stress, pd.DataFrame()
    numeric = stress.select_dtypes(include=[np.number])
    summary_rows = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        summary_rows.append(
            {
                "metric": column,
                "mean": float(values.mean()),
                "median": float(values.median()),
                "p05": float(values.quantile(0.05)),
                "p25": float(values.quantile(0.25)),
                "p75": float(values.quantile(0.75)),
                "p95": float(values.quantile(0.95)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return stress, pd.DataFrame(summary_rows)


def make_state_stability_row(
    window_id: int,
    previous_fit: HMMFitResult | None,
    current_fit: HMMFitResult,
) -> dict[str, Any]:
    current_map = {int(k): v for k, v in current_fit.state_map.items()}
    previous_map = (
        {} if previous_fit is None else {int(k): v for k, v in previous_fit.state_map.items()}
    )
    row: dict[str, Any] = {
        "window_id": window_id,
        "previous_state_map": json.dumps(previous_map, sort_keys=True),
        "current_state_map": json.dumps(current_map, sort_keys=True),
        "map_changed": bool(previous_map and previous_map != current_map),
        "bull_component_changed": False,
        "bear_component_changed": False,
        "side_component_changed": False,
        "transition_matrix_distance": np.nan,
        "emission_mean_distance": np.nan,
    }
    if previous_fit is None:
        return row

    for state, output_name in {
        STATE_BULL: "bull_component_changed",
        STATE_BEAR: "bear_component_changed",
        STATE_SIDE: "side_component_changed",
    }.items():
        row[output_name] = _state_components(previous_fit.state_map, state) != _state_components(
            current_fit.state_map,
            state,
        )

    previous_trans = np.asarray(previous_fit.model.transmat_, dtype=float)
    current_trans = np.asarray(current_fit.model.transmat_, dtype=float)
    if previous_trans.shape == current_trans.shape:
        row["transition_matrix_distance"] = _safe_float(
            np.linalg.norm(current_trans - previous_trans)
        )

    previous_means = np.asarray(previous_fit.model.means_, dtype=float)
    current_means = np.asarray(current_fit.model.means_, dtype=float)
    if previous_means.shape == current_means.shape:
        row["emission_mean_distance"] = _safe_float(np.linalg.norm(current_means - previous_means))

    return row


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
            "avg_asset_return_when_decoded": _safe_float(
                backtest.loc[decoded_mask, "ret_1"].mean()
            ),
            "avg_strategy_return_when_decoded": _safe_float(
                backtest.loc[decoded_mask, "strategy_ret"].mean()
            ),
        }

        if not window_diagnostics.empty:
            dwell_col = f"{state}_expected_dwell"
            fwd_col = f"{state}_forward_return"
            occ_col = f"{state}_occupancy"
            row["avg_train_expected_dwell"] = _safe_float(
                window_diagnostics.get(dwell_col, pd.Series(dtype=float)).mean()
            )
            row["avg_train_forward_return"] = _safe_float(
                window_diagnostics.get(fwd_col, pd.Series(dtype=float)).mean()
            )
            row["avg_train_occupancy"] = _safe_float(
                window_diagnostics.get(occ_col, pd.Series(dtype=float)).mean()
            )
        else:
            row["avg_train_expected_dwell"] = np.nan
            row["avg_train_forward_return"] = np.nan
            row["avg_train_occupancy"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)
