from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .diagnostics import (
    make_component_diagnostic_rows,
    make_decode_comparison_rows,
    make_emission_diagnostic_rows,
    make_seed_diagnostic_rows,
    make_state_stability_row,
    make_stationary_distribution_rows,
    make_synthetic_sample_stress,
    make_transition_matrix_row,
    make_window_diagnostic_row,
)
from .ensemble import add_ensemble_signals
from .features import feature_columns
from .hmm_model import fit_hmm_best_of_seeds, semantic_probabilities
from .nn_model import add_nn_confirmation
from .states import STATE_BEAR, STATE_BULL, STATE_ORDER, STATE_SIDE


@dataclass
class BacktestResult:
    backtest: pd.DataFrame
    window_diagnostics: pd.DataFrame
    seed_diagnostics: pd.DataFrame
    hmm_model_selection: pd.DataFrame
    hmm_emissions: pd.DataFrame
    hmm_transition_matrices: pd.DataFrame
    hmm_state_stability: pd.DataFrame
    hmm_stationary_distribution: pd.DataFrame
    hmm_decode_comparison: pd.DataFrame
    hmm_component_diagnostics: pd.DataFrame
    hmm_synthetic_sample_stress: pd.DataFrame
    hmm_synthetic_sample_stress_summary: pd.DataFrame


def drawdown_multiplier(current_dd: float, cfg: PipelineConfig) -> float:
    if not cfg.use_drawdown_throttle:
        return 1.0

    dd_abs = abs(min(current_dd, 0.0))

    if dd_abs >= cfg.dd_level_2:
        return cfg.dd_mult_2
    if dd_abs >= cfg.dd_level_1:
        return cfg.dd_mult_1

    return 1.0


def compute_position_for_row(row: pd.Series, equity_dd: float, cfg: PipelineConfig) -> float:
    bull = float(row[STATE_BULL])
    bear = float(row[STATE_BEAR])
    side = float(row[STATE_SIDE])
    signal = float(row.get("ensemble_signal", bull - bear))

    dominant = max(
        [(bull, STATE_BULL), (bear, STATE_BEAR), (side, STATE_SIDE)],
        key=lambda x: x[0],
    )[1]

    if dominant == STATE_BULL:
        regime_mult = cfg.bull_regime_exposure_mult
    elif dominant == STATE_SIDE:
        regime_mult = cfg.side_regime_exposure_mult
    else:
        regime_mult = cfg.bear_regime_exposure_mult

    if cfg.use_vol_targeting and row["rv_realised"] > 0:
        vol_mult = np.clip(
            cfg.target_ann_vol / row["rv_realised"], cfg.min_vol_mult, cfg.max_vol_mult
        )
    else:
        vol_mult = 1.0

    dd_mult = drawdown_multiplier(equity_dd, cfg)
    confidence_mult = np.clip(abs(signal) / max(cfg.min_signal, 1e-9), 0.25, 1.50)

    alpha_long = row["alpha_score_long"] >= cfg.min_alpha_score
    alpha_short = row["alpha_score_short"] >= cfg.min_alpha_score

    core_long = (
        bool(row["trend_long"]) and bool(row["rs_long"]) and bear <= cfg.max_bear_prob_for_long
    )

    adaptive_long = (
        bull >= cfg.min_bull_prob
        and bear <= cfg.max_bear_prob_for_long
        and signal >= cfg.min_signal
        and alpha_long
    )

    long_exposure = 0.0
    if core_long:
        long_exposure = max(long_exposure, cfg.core_exposure * vol_mult * dd_mult)
    if adaptive_long:
        long_exposure = max(
            long_exposure,
            cfg.base_exposure * confidence_mult * vol_mult * dd_mult * regime_mult,
        )

    short_exposure = 0.0
    if not cfg.long_only:
        adaptive_short = bear >= cfg.min_bull_prob and signal <= -cfg.min_signal and alpha_short
        if adaptive_short:
            short_exposure = -cfg.base_exposure * confidence_mult * vol_mult * dd_mult

    target = long_exposure if abs(long_exposure) >= abs(short_exposure) else short_exposure
    return float(np.clip(target, -cfg.max_exposure, cfg.max_exposure))


def _semantic_confidence(next_probs: pd.DataFrame) -> pd.DataFrame:
    probs = next_probs[STATE_ORDER].astype(float).clip(lower=0.0)
    row_sums = probs.sum(axis=1).replace(0.0, np.nan)
    normalized = probs.div(row_sums, axis=0).fillna(0.0)
    entropy = -(normalized * np.log(normalized.replace(0.0, np.nan))).sum(axis=1).fillna(0.0)
    denominator = np.log(len(STATE_ORDER))
    sorted_probs = np.sort(normalized.values, axis=1)
    top = sorted_probs[:, -1]
    second = sorted_probs[:, -2] if normalized.shape[1] > 1 else np.zeros(len(normalized))
    return pd.DataFrame(
        {
            "hmm_posterior_entropy": entropy.values,
            "hmm_max_posterior": top,
            "hmm_confidence": np.clip(1.0 - entropy.values / denominator, 0.0, 1.0),
            "hmm_state_margin": top - second,
        },
        index=next_probs.index,
    )


def walk_forward_backtest(df: pd.DataFrame, cfg: PipelineConfig) -> BacktestResult:
    features = feature_columns(cfg)
    df = df.copy()
    window_rows = []
    seed_rows = []
    model_selection_rows = []
    emission_rows = []
    transition_rows = []
    stability_rows = []
    stationary_rows = []
    decode_rows = []
    component_diagnostic_rows = []
    fits_for_sample = []
    previous_fit = None

    for col in STATE_ORDER:
        df[col] = np.nan

    df["hidden_state"] = pd.Series(index=df.index, dtype="object")
    df["decoded_state_viterbi"] = pd.Series(index=df.index, dtype="object")
    df["decoded_state_map"] = pd.Series(index=df.index, dtype="object")
    df["state_decode_disagreement"] = False
    df["hmm_window_id"] = np.nan
    df["hmm_log_likelihood"] = np.nan
    df["hmm_window_log_likelihood"] = np.nan
    df["hmm_posterior_entropy"] = np.nan
    df["hmm_max_posterior"] = np.nan
    df["hmm_confidence"] = np.nan
    df["hmm_state_margin"] = np.nan
    df["target_position"] = 0.0

    start_idx = cfg.train_window
    end_idx = len(df) - 1

    if end_idx <= start_idx:
        raise ValueError(
            f"Not enough data after features. Need more than train_window={cfg.train_window}, "
            f"but only have {len(df)} rows."
        )

    for window_id, train_end in enumerate(
        range(start_idx, end_idx, cfg.retrain_frequency), start=1
    ):
        test_end = min(train_end + cfg.retrain_frequency, end_idx)

        train = df.iloc[train_end - cfg.train_window : train_end].copy()
        test = df.iloc[train_end:test_end].copy()

        try:
            fit = fit_hmm_best_of_seeds(train, features, cfg)
        except RuntimeError as err:
            print(f"Skipping window ending {df.index[train_end].date()}: {err}")
            continue

        model = fit.model
        X_test = fit.scaler.transform(test[features].values)

        test_log_prob, posteriors_raw = model.score_samples(X_test)
        next_probs_raw = posteriors_raw @ model.transmat_

        next_probs = semantic_probabilities(next_probs_raw, fit.state_map)

        _, decoded_raw = model.decode(X_test, algorithm="viterbi")
        _, decoded_map_raw = model.decode(X_test, algorithm="map")
        decoded = [fit.state_map[int(x)] for x in decoded_raw]
        decoded_map = [fit.state_map[int(x)] for x in decoded_map_raw]

        idx = test.index
        confidence = _semantic_confidence(next_probs)
        df.loc[idx, STATE_ORDER] = next_probs[STATE_ORDER].values
        df.loc[idx, "hidden_state"] = decoded
        df.loc[idx, "decoded_state_viterbi"] = decoded
        df.loc[idx, "decoded_state_map"] = decoded_map
        df.loc[idx, "state_decode_disagreement"] = np.asarray(decoded) != np.asarray(decoded_map)
        df.loc[idx, "hmm_window_id"] = window_id
        df.loc[idx, "hmm_window_log_likelihood"] = float(test_log_prob)
        df.loc[idx, "hmm_log_likelihood"] = np.nan
        df.loc[idx, confidence.columns] = confidence.values

        if cfg.enable_diagnostics:
            window_row = make_window_diagnostic_row(window_id, train, test, fit, next_probs)
            window_row["test_log_likelihood"] = float(test_log_prob)
            window_row["decode_disagreement_rate"] = float(np.mean(decoded_raw != decoded_map_raw))
            window_rows.append(window_row)
            seed_rows.extend(make_seed_diagnostic_rows(window_id, train.index[-1], fit))
            for row in fit.model_selection_rows:
                model_selection_rows.append({"window_id": window_id, **row})
            emission_rows.extend(make_emission_diagnostic_rows(window_id, train, fit, features))
            transition_rows.append(make_transition_matrix_row(window_id, train, test, fit))
            stationary_rows.extend(make_stationary_distribution_rows(window_id, fit))
            decode_rows.extend(
                make_decode_comparison_rows(
                    window_id,
                    test.index,
                    decoded_raw,
                    decoded_map_raw,
                    next_probs,
                    fit,
                )
            )
            component_diagnostic_rows.extend(make_component_diagnostic_rows(window_id, train, fit))
            stability_rows.append(make_state_stability_row(window_id, previous_fit, fit))
            fits_for_sample.append((window_id, fit))
            previous_fit = fit

    df["hmm_signal"] = df[STATE_BULL] - df[STATE_BEAR]
    if cfg.enable_nn_confirmation or cfg.ensemble_mode != "hmm_only":
        df = add_nn_confirmation(df, features, cfg)
        df = add_ensemble_signals(df, cfg)

    equity = 1.0
    peak = 1.0
    positions = []
    equity_curve = []

    cost_rate = (cfg.commission_bps + cfg.slippage_bps) / 10_000.0
    prev_pos = 0.0

    for _, row in df.iterrows():
        if np.isnan(row[STATE_BULL]) or np.isnan(row["rv_realised"]):
            target_pos = 0.0
        else:
            current_dd = equity / peak - 1.0
            target_pos = compute_position_for_row(row, current_dd, cfg)

        asset_ret = row["ret_1"] if np.isfinite(row["ret_1"]) else 0.0

        turnover = abs(target_pos - prev_pos)
        strategy_ret = prev_pos * asset_ret - turnover * cost_rate

        equity *= 1.0 + strategy_ret
        peak = max(peak, equity)

        positions.append(target_pos)
        equity_curve.append(equity)
        prev_pos = target_pos

    df["target_position"] = positions
    df["position"] = positions
    df["strategy_equity"] = equity_curve
    df["strategy_ret"] = df["strategy_equity"].pct_change().fillna(0.0)
    df["buy_hold_equity"] = (1.0 + df["ret_1"].fillna(0.0)).cumprod()

    sample_stress, sample_stress_summary = (
        make_synthetic_sample_stress(fits_for_sample, cfg)
        if cfg.enable_diagnostics
        else (pd.DataFrame(), pd.DataFrame())
    )

    return BacktestResult(
        backtest=df,
        window_diagnostics=pd.DataFrame(window_rows),
        seed_diagnostics=pd.DataFrame(seed_rows),
        hmm_model_selection=pd.DataFrame(model_selection_rows),
        hmm_emissions=pd.DataFrame(emission_rows),
        hmm_transition_matrices=pd.DataFrame(transition_rows),
        hmm_state_stability=pd.DataFrame(stability_rows),
        hmm_stationary_distribution=pd.DataFrame(stationary_rows),
        hmm_decode_comparison=pd.DataFrame(decode_rows),
        hmm_component_diagnostics=pd.DataFrame(component_diagnostic_rows),
        hmm_synthetic_sample_stress=sample_stress,
        hmm_synthetic_sample_stress_summary=sample_stress_summary,
    )
