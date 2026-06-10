from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM, GaussianHMM
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
    convergence_history: tuple[float, ...] = ()
    final_likelihood_delta: float = np.nan


@dataclass
class HMMFitResult:
    model: GaussianHMM | GMMHMM
    scaler: StandardScaler
    state_map: dict[int, str]
    best_seed: int
    train_log_likelihood: float
    seed_diagnostics: list[SeedFitDiagnostic]
    component_forward_returns: dict[int, float]
    component_occupancy: dict[int, float]
    component_diagnostics: dict[int, dict[str, float]] = field(default_factory=dict)
    model_selection_rows: list[dict[str, object]] = field(default_factory=list)
    selected_n_states: int = 3
    selected_covariance_type: str = "full"
    selected_model_type: str = "gaussian"
    selected_n_mix: int = 1
    selected_min_covar: float = 1e-3
    selected_transmat_prior: float = 1.0
    feature_names: list[str] = field(default_factory=list)
    train_aic: float = np.nan
    train_bic: float = np.nan
    train_ic_source: str = ""
    convergence_history: tuple[float, ...] = ()
    final_likelihood_delta: float = np.nan


def convergence_history(model: GaussianHMM | GMMHMM) -> tuple[float, ...]:
    history = getattr(getattr(model, "monitor_", None), "history", ())
    return tuple(float(value) for value in history)


def final_likelihood_delta(model: GaussianHMM | GMMHMM) -> float:
    history = convergence_history(model)
    if len(history) < 2:
        return np.nan
    return float(history[-1] - history[-2])


def model_information_criteria(
    model: GaussianHMM | GMMHMM,
    X: np.ndarray,
    n_parameters: int | None = None,
    use_native: bool = True,
) -> tuple[float, float, str]:
    if use_native and hasattr(model, "aic") and hasattr(model, "bic"):
        try:
            return float(model.aic(X)), float(model.bic(X)), "hmmlearn_native"
        except Exception:
            pass

    if n_parameters is None or X.shape[0] <= 0:
        return np.nan, np.nan, "unavailable"

    log_likelihood = float(model.score(X))
    aic = 2.0 * n_parameters - 2.0 * log_likelihood
    bic = np.log(max(X.shape[0], 1)) * n_parameters - 2.0 * log_likelihood
    return float(aic), float(bic), "local_fallback"


def build_hmm(
    cfg: PipelineConfig,
    random_state: int,
    candidate: dict[str, object] | None = None,
) -> GaussianHMM | GMMHMM:
    candidate = candidate or {}
    model_type = str(candidate.get("model_type", cfg.hmm_model_type)).lower()
    n_components = int(candidate.get("n_states", cfg.n_states))
    covariance_type = str(candidate.get("covariance_type", cfg.covariance_type))
    min_covar = float(candidate.get("min_covar", cfg.hmm_min_covar))
    transmat_prior = float(candidate.get("transmat_prior", cfg.hmm_transmat_prior))
    common = {
        "n_components": n_components,
        "min_covar": min_covar,
        "startprob_prior": cfg.hmm_startprob_prior,
        "transmat_prior": transmat_prior,
        "algorithm": cfg.hmm_decode_algorithm,
        "random_state": int(random_state),
        "n_iter": cfg.n_iter,
        "tol": cfg.tolerance,
        "params": str(candidate.get("params", cfg.hmm_params)),
        "init_params": str(candidate.get("init_params", cfg.hmm_init_params)),
        "implementation": cfg.hmm_implementation,
    }

    if model_type == "gmm":
        return GMMHMM(
            n_mix=int(candidate.get("n_mix", cfg.hmm_n_mix)),
            covariance_type=covariance_type,
            weights_prior=cfg.hmm_weights_prior,
            means_prior=cfg.hmm_means_prior,
            means_weight=cfg.hmm_means_weight,
            covars_prior=cfg.hmm_covars_prior,
            covars_weight=cfg.hmm_covars_weight,
            params=str(candidate.get("params", cfg.hmm_params_gmm)),
            init_params=str(candidate.get("init_params", cfg.hmm_init_params_gmm)),
            **{k: v for k, v in common.items() if k not in {"params", "init_params"}},
        )

    if model_type != "gaussian":
        raise ValueError(f"Unsupported hmm_model_type={model_type}")

    return GaussianHMM(
        covariance_type=covariance_type,
        means_prior=cfg.hmm_means_prior,
        means_weight=cfg.hmm_means_weight,
        covars_prior=cfg.hmm_covars_prior,
        covars_weight=cfg.hmm_covars_weight,
        **common,
    )


def fit_hmm_best_of_seeds(
    train_df: pd.DataFrame,
    features: list[str],
    cfg: PipelineConfig,
) -> HMMFitResult:
    if cfg.enable_hmm_model_selection:
        from .model_selection import select_hmm_model

        return select_hmm_model(train_df, features, cfg)

    scaler = StandardScaler()
    X = scaler.fit_transform(train_df[features].values)

    best_model = None
    best_score = -np.inf
    best_seed = -1
    seed_diagnostics: list[SeedFitDiagnostic] = []

    for seed in cfg.random_seeds:
        try:
            model = build_hmm(cfg, int(seed))
            model.fit(X)
            score = model.score(X)
            converged = bool(getattr(model.monitor_, "converged", False))
            iterations = int(getattr(model.monitor_, "iter", 0))
            history = convergence_history(model)

            seed_diagnostics.append(
                SeedFitDiagnostic(
                    seed=int(seed),
                    score=float(score),
                    converged=converged,
                    iterations=iterations,
                    convergence_history=history if cfg.hmm_store_convergence_history else (),
                    final_likelihood_delta=final_likelihood_delta(model),
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
    state_map, component_diagnostics = map_components_to_semantic_states(
        best_model,
        train_df,
        hidden_train,
        cfg,
    )
    component_scores = {
        component: values["forward_return"] for component, values in component_diagnostics.items()
    }
    component_occupancy = {
        component: values["occupancy"] for component, values in component_diagnostics.items()
    }
    train_aic, train_bic, ic_source = model_information_criteria(
        best_model,
        X,
        use_native=cfg.hmm_use_native_information_criteria,
    )
    best_history = convergence_history(best_model)

    return HMMFitResult(
        model=best_model,
        scaler=scaler,
        state_map=state_map,
        best_seed=best_seed,
        train_log_likelihood=float(best_score),
        seed_diagnostics=seed_diagnostics,
        component_forward_returns=component_scores,
        component_occupancy=component_occupancy,
        component_diagnostics=component_diagnostics,
        selected_n_states=cfg.n_states,
        selected_covariance_type=cfg.covariance_type,
        selected_model_type=cfg.hmm_model_type,
        selected_n_mix=cfg.hmm_n_mix if cfg.hmm_model_type == "gmm" else 1,
        selected_min_covar=cfg.hmm_min_covar,
        selected_transmat_prior=cfg.hmm_transmat_prior,
        feature_names=list(features),
        train_aic=train_aic,
        train_bic=train_bic,
        train_ic_source=ic_source,
        convergence_history=best_history if cfg.hmm_store_convergence_history else (),
        final_likelihood_delta=final_likelihood_delta(best_model),
    )


def component_mean(model: GaussianHMM | GMMHMM, component: int) -> np.ndarray:
    means = np.asarray(model.means_, dtype=float)
    if means.ndim == 3:
        weights = np.asarray(model.weights_[component], dtype=float)
        return np.average(means[component], axis=0, weights=weights)
    return means[component]


def component_covariance(model: GaussianHMM | GMMHMM, component: int) -> np.ndarray:
    covars = np.asarray(model.covars_, dtype=float)
    if covars.ndim == 4:
        weights = np.asarray(model.weights_[component], dtype=float)
        return np.average(covars[component], axis=0, weights=weights)
    if covars.ndim == 3:
        item = covars[component]
        if hasattr(model, "weights_") and item.ndim == 2:
            weights = np.asarray(model.weights_[component], dtype=float)
            return np.diag(np.average(item, axis=0, weights=weights))
        if item.ndim == 2:
            return item
        return np.diag(np.asarray(item, dtype=float))
    if covars.ndim == 2:
        if hasattr(model, "weights_"):
            weights = np.asarray(model.weights_[component], dtype=float)
            n_features = int(np.asarray(model.means_).shape[-1])
            return np.diag(np.repeat(np.average(covars[component], weights=weights), n_features))
        return np.diag(covars[component])
    return np.diag(np.atleast_1d(covars[component]))


def _component_drawdown(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def map_components_to_semantic_states(
    model: GaussianHMM,
    train_df: pd.DataFrame,
    hidden_states: np.ndarray,
    cfg: PipelineConfig,
) -> tuple[dict[int, str], dict[int, dict[str, float]]]:
    """Map raw HMM components into Bull/Bear/Sideways semantic states."""
    fwd_ret = train_df["ret_1"].shift(-1)
    min_occupancy = max(float(cfg.semantic_min_occupancy), 0.0)
    diagnostics: dict[int, dict[str, float]] = {}

    for component in range(model.n_components):
        mask = hidden_states == component
        component_returns = fwd_ret[mask].dropna()
        trend = (
            train_df.loc[mask, "alpha_score_long"].mean()
            if "alpha_score_long" in train_df
            else np.nan
        )
        forward_return = float(component_returns.mean()) if not component_returns.empty else np.nan
        volatility = float(component_returns.std(ddof=0)) if len(component_returns) > 1 else np.nan
        drawdown = _component_drawdown(component_returns)
        skew = float(component_returns.skew()) if len(component_returns) > 2 else np.nan
        occupancy = float(mask.mean())
        trend_score = float(trend) if np.isfinite(trend) else 0.0
        safe_forward = forward_return if np.isfinite(forward_return) else -np.inf
        safe_vol = volatility if np.isfinite(volatility) else 0.0
        safe_drawdown = drawdown if np.isfinite(drawdown) else 0.0
        semantic_score = safe_forward - 0.25 * safe_vol + 0.05 * trend_score + 0.10 * safe_drawdown
        bull_score = safe_forward + 0.5 * trend_score - 0.5 * safe_vol + 0.25 * safe_drawdown
        bear_score = -safe_forward + 0.5 * safe_vol - 0.25 * trend_score - 0.25 * safe_drawdown
        sideways_score = -abs(safe_forward) - 0.25 * abs(trend_score) + 0.1 * safe_vol

        diagnostics[component] = {
            "forward_return": forward_return,
            "volatility": volatility,
            "drawdown": drawdown,
            "occupancy": occupancy,
            "trend_score": trend_score,
            "skew": skew,
            "semantic_score": float(semantic_score),
            "bull_score": float(bull_score),
            "bear_score": float(bear_score),
            "sideways_score": float(sideways_score),
        }

    components = list(diagnostics)
    viable = [c for c in components if diagnostics[c]["occupancy"] >= min_occupancy] or components

    bull_component = max(viable, key=lambda c: diagnostics[c]["semantic_score"])
    bear_component = min(viable, key=lambda c: diagnostics[c]["semantic_score"])
    if bull_component == bear_component and len(viable) > 1:
        bear_component = min(
            [c for c in viable if c != bull_component],
            key=lambda c: diagnostics[c]["forward_return"],
        )

    finite_returns = np.array(
        [
            diagnostics[c]["forward_return"]
            for c in viable
            if np.isfinite(diagnostics[c]["forward_return"])
        ],
        dtype=float,
    )
    high_cut = np.quantile(finite_returns, 0.60) if finite_returns.size else np.inf
    low_cut = np.quantile(finite_returns, 0.40) if finite_returns.size else -np.inf

    state_map: dict[int, str] = {}
    for component in components:
        fwd = diagnostics[component]["forward_return"]
        occupancy = diagnostics[component]["occupancy"]
        if component == bull_component or (
            model.n_components > 3
            and occupancy >= min_occupancy
            and np.isfinite(fwd)
            and fwd >= high_cut
            and fwd > 0.0
        ):
            state_map[component] = STATE_BULL
        elif component == bear_component or (
            model.n_components > 3
            and occupancy >= min_occupancy
            and np.isfinite(fwd)
            and fwd <= low_cut
            and fwd < 0.0
        ):
            state_map[component] = STATE_BEAR
        else:
            state_map[component] = STATE_SIDE

    return state_map, diagnostics


def semantic_probabilities(probs_raw: np.ndarray, state_map: dict[int, str]) -> pd.DataFrame:
    out = pd.DataFrame(0.0, index=np.arange(len(probs_raw)), columns=STATE_ORDER)

    for component, semantic_state in state_map.items():
        out[semantic_state] += probs_raw[:, component]

    return out
