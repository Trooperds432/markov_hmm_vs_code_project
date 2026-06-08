from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .hmm_model import (
    HMMFitResult,
    SeedFitDiagnostic,
    map_components_to_semantic_states,
    semantic_probabilities,
)
from .states import STATE_BEAR, STATE_BULL


@dataclass
class HMMCandidateFit:
    model: GaussianHMM | None
    scaler: StandardScaler | None
    row: dict[str, object]
    state_map: dict[int, str] | None
    component_diagnostics: dict[int, dict[str, float]]


def validation_split(
    train_df: pd.DataFrame,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fraction = float(np.clip(validation_fraction, 0.05, 0.50))
    split_idx = int(round(len(train_df) * (1.0 - fraction)))
    split_idx = min(max(split_idx, 20), len(train_df) - 5)
    return train_df.iloc[:split_idx].copy(), train_df.iloc[split_idx:].copy()


def hmm_num_parameters(n_states: int, n_features: int, covariance_type: str) -> int:
    start_probs = n_states - 1
    transition_probs = n_states * (n_states - 1)
    means = n_states * n_features
    if covariance_type == "full":
        covariances = n_states * n_features * (n_features + 1) // 2
    else:
        covariances = n_states * n_features
    return int(start_probs + transition_probs + means + covariances)


def information_criteria(
    log_likelihood: float, n_parameters: int, n_obs: int
) -> tuple[float, float]:
    if not np.isfinite(log_likelihood) or n_obs <= 0:
        return np.nan, np.nan
    aic = 2.0 * n_parameters - 2.0 * log_likelihood
    bic = np.log(max(n_obs, 1)) * n_parameters - 2.0 * log_likelihood
    return float(aic), float(bic)


def _score_candidate_row(row: dict[str, object], metric: str) -> float:
    value = float(row.get(metric, np.nan))
    if not np.isfinite(value):
        return -np.inf
    if metric in {"validation_aic", "validation_bic", "aic", "bic"}:
        return -value
    return value


def _validation_sharpe_proxy(
    model: GaussianHMM,
    scaler: StandardScaler,
    validation: pd.DataFrame,
    features: list[str],
    state_map: dict[int, str],
) -> float:
    X_val = scaler.transform(validation[features].values)
    probs = semantic_probabilities(model.predict_proba(X_val) @ model.transmat_, state_map)
    signal = (probs[STATE_BULL] - probs[STATE_BEAR]).clip(-1.0, 1.0)
    proxy_returns = (
        signal.shift(1).fillna(0.0).to_numpy() * validation["ret_1"].fillna(0.0).to_numpy()
    )
    std = float(np.std(proxy_returns))
    if std == 0.0 or np.isnan(std):
        return np.nan
    return float(np.mean(proxy_returns) / std * np.sqrt(252))


def fit_hmm_candidate(
    inner_train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    n_states: int,
    covariance_type: str,
    seed: int,
    cfg: PipelineConfig,
) -> HMMCandidateFit:
    row: dict[str, object] = {
        "n_states": int(n_states),
        "covariance_type": covariance_type,
        "seed": int(seed),
        "converged": False,
        "iterations": 0,
        "train_log_likelihood": np.nan,
        "validation_log_likelihood": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "validation_aic": np.nan,
        "validation_bic": np.nan,
        "validation_sharpe_proxy": np.nan,
        "selected": False,
        "error": "",
    }

    try:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(inner_train[features].values)
        X_val = scaler.transform(validation[features].values)
        model = GaussianHMM(
            n_components=int(n_states),
            covariance_type=covariance_type,
            n_iter=cfg.n_iter,
            tol=cfg.tolerance,
            random_state=int(seed),
            algorithm="viterbi",
            implementation="scaling",
        )
        model.fit(X_train)
        train_ll = float(model.score(X_train))
        validation_ll = float(model.score(X_val))
        n_params = hmm_num_parameters(int(n_states), len(features), covariance_type)
        aic, bic = information_criteria(validation_ll, n_params, len(validation))
        hidden_train = model.predict(X_train)
        state_map, component_diagnostics = map_components_to_semantic_states(
            model,
            inner_train,
            hidden_train,
            cfg,
        )
        row.update(
            {
                "converged": bool(getattr(model.monitor_, "converged", False)),
                "iterations": int(getattr(model.monitor_, "iter", 0)),
                "train_log_likelihood": train_ll,
                "validation_log_likelihood": validation_ll,
                "aic": aic,
                "bic": bic,
                "validation_aic": aic,
                "validation_bic": bic,
                "validation_sharpe_proxy": _validation_sharpe_proxy(
                    model,
                    scaler,
                    validation,
                    features,
                    state_map,
                ),
            }
        )
        return HMMCandidateFit(model, scaler, row, state_map, component_diagnostics)
    except Exception as err:
        row["error"] = type(err).__name__
        return HMMCandidateFit(None, None, row, None, {})


def _refit_selected(
    train_df: pd.DataFrame,
    features: list[str],
    selected_row: dict[str, object],
    cfg: PipelineConfig,
    fallback: HMMCandidateFit,
) -> tuple[GaussianHMM, StandardScaler, float]:
    if not cfg.refit_selected_hmm_on_full_train:
        if fallback.model is None or fallback.scaler is None:
            raise RuntimeError("Selected HMM candidate is unavailable.")
        return fallback.model, fallback.scaler, float(fallback.row["train_log_likelihood"])

    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[features].values)
    model = GaussianHMM(
        n_components=int(selected_row["n_states"]),
        covariance_type=str(selected_row["covariance_type"]),
        n_iter=cfg.n_iter,
        tol=cfg.tolerance,
        random_state=int(selected_row["seed"]),
        algorithm="viterbi",
        implementation="scaling",
    )
    model.fit(X)
    return model, scaler, float(model.score(X))


def select_hmm_model(
    train_df: pd.DataFrame,
    features: list[str],
    cfg: PipelineConfig,
) -> HMMFitResult:
    inner_train, validation = validation_split(train_df, cfg.hmm_validation_fraction)
    candidates: list[HMMCandidateFit] = []

    for n_states in cfg.hmm_state_candidates:
        for covariance_type in cfg.hmm_covariance_candidates:
            for seed in cfg.random_seeds:
                candidates.append(
                    fit_hmm_candidate(
                        inner_train,
                        validation,
                        features,
                        int(n_states),
                        str(covariance_type),
                        int(seed),
                        cfg,
                    )
                )

    valid = [candidate for candidate in candidates if candidate.model is not None]
    if not valid:
        raise RuntimeError("All HMM model-selection candidates failed.")

    metric = cfg.hmm_model_selection_metric
    if metric == "validation_aic":
        metric = "validation_aic"
    elif metric == "validation_bic":
        metric = "validation_bic"
    elif metric not in {
        "train_log_likelihood",
        "validation_log_likelihood",
        "validation_sharpe_proxy",
    }:
        metric = "validation_bic"

    selected = max(valid, key=lambda candidate: _score_candidate_row(candidate.row, metric))
    selected.row["selected"] = True
    rows = [candidate.row for candidate in candidates]

    model, scaler, train_ll = _refit_selected(train_df, features, selected.row, cfg, selected)
    X_full = scaler.transform(train_df[features].values)
    hidden_train = model.predict(X_full)
    state_map, component_diagnostics = map_components_to_semantic_states(
        model,
        train_df,
        hidden_train,
        cfg,
    )
    component_forward_returns = {
        component: values["forward_return"] for component, values in component_diagnostics.items()
    }
    component_occupancy = {
        component: values["occupancy"] for component, values in component_diagnostics.items()
    }

    seed_diagnostics = [
        SeedFitDiagnostic(
            seed=int(row["seed"]),
            score=float(row["train_log_likelihood"]),
            converged=bool(row["converged"]),
            iterations=int(row["iterations"]),
            error=str(row["error"]),
        )
        for row in rows
        if int(row["n_states"]) == int(selected.row["n_states"])
        and str(row["covariance_type"]) == str(selected.row["covariance_type"])
    ]

    return HMMFitResult(
        model=model,
        scaler=scaler,
        state_map=state_map,
        best_seed=int(selected.row["seed"]),
        train_log_likelihood=train_ll,
        seed_diagnostics=seed_diagnostics,
        component_forward_returns=component_forward_returns,
        component_occupancy=component_occupancy,
        component_diagnostics=component_diagnostics,
        model_selection_rows=rows,
        selected_n_states=int(selected.row["n_states"]),
        selected_covariance_type=str(selected.row["covariance_type"]),
    )
