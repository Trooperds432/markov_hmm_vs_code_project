from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .states import STATE_BEAR, STATE_BULL, STATE_ORDER, STATE_SIDE


@dataclass
class SeedFitDiagnostic:
    seed: int
    score: float
    converged: bool
    iterations: int
    error: str = ""


@dataclass
class HMMFitResult:
    model: GaussianHMM
    scaler: StandardScaler
    state_map: Dict[int, str]
    best_seed: int
    train_log_likelihood: float
    seed_diagnostics: list[SeedFitDiagnostic]
    component_forward_returns: Dict[int, float]
    component_occupancy: Dict[int, float]


def fit_hmm_best_of_seeds(
    train_df: pd.DataFrame,
    features: list[str],
    cfg: PipelineConfig,
) -> HMMFitResult:
    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[features].values)

    best_model = None
    best_score = -np.inf
    best_seed = -1
    seed_diagnostics: list[SeedFitDiagnostic] = []

    for seed in cfg.random_seeds:
        try:
            model = GaussianHMM(
                n_components=cfg.n_states,
                covariance_type=cfg.covariance_type,
                n_iter=cfg.n_iter,
                tol=cfg.tolerance,
                random_state=seed,
                algorithm="viterbi",
                implementation="scaling",
            )
            model.fit(X)
            score = model.score(X)
            converged = bool(getattr(model.monitor_, "converged", False))
            iterations = int(getattr(model.monitor_, "iter", 0))

            seed_diagnostics.append(
                SeedFitDiagnostic(
                    seed=int(seed),
                    score=float(score),
                    converged=converged,
                    iterations=iterations,
                )
            )

            if np.isfinite(score) and score > best_score:
                best_score = score
                best_model = model
                best_seed = int(seed)
        except Exception:
            seed_diagnostics.append(
                SeedFitDiagnostic(
                    seed=int(seed),
                    score=np.nan,
                    converged=False,
                    iterations=0,
                    error="fit_failed",
                )
            )
            continue

    if best_model is None:
        raise RuntimeError("All HMM fits failed. Try covariance_type='diag' or reduce features.")

    hidden_train = best_model.predict(X)
    fwd_ret = train_df["ret_1"].shift(-1)

    component_scores = {}
    component_occupancy = {}
    for component in range(cfg.n_states):
        mask = hidden_train == component
        component_scores[component] = float(fwd_ret[mask].mean())
        component_occupancy[component] = float(mask.mean())

    def sort_score(component: int) -> float:
        score = component_scores[component]
        return score if np.isfinite(score) else -np.inf

    sorted_components = sorted(component_scores, key=sort_score, reverse=True)

    state_map = {
        sorted_components[0]: STATE_BULL,
        sorted_components[-1]: STATE_BEAR,
    }

    for component in range(cfg.n_states):
        if component not in state_map:
            state_map[component] = STATE_SIDE

    return HMMFitResult(
        model=best_model,
        scaler=scaler,
        state_map=state_map,
        best_seed=best_seed,
        train_log_likelihood=float(best_score),
        seed_diagnostics=seed_diagnostics,
        component_forward_returns=component_scores,
        component_occupancy=component_occupancy,
    )


def semantic_probabilities(probs_raw: np.ndarray, state_map: Dict[int, str]) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=np.arange(len(probs_raw)), columns=STATE_ORDER)

    for component, semantic_state in state_map.items():
        out[semantic_state] += probs_raw[:, component]

    return out
