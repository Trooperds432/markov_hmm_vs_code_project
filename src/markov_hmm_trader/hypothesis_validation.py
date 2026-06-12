from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .metrics import max_drawdown, sharpe
from .states import STATE_BEAR, STATE_BULL

MIN_REQUIRED_SAMPLE = 30


def _forward_return(df: pd.DataFrame, horizon: int = 20) -> pd.Series:
    return (
        df.get("ret_1", pd.Series(index=df.index, dtype=float))
        .shift(-horizon)
        .rolling(
            horizon,
            min_periods=1,
        )
        .sum()
    )


def _ttest_pvalue(a: pd.Series, b: pd.Series) -> float:
    if importlib.util.find_spec("scipy") is None:
        return np.nan
    from scipy import stats

    a_clean = a.replace([np.inf, -np.inf], np.nan).dropna()
    b_clean = b.replace([np.inf, -np.inf], np.nan).dropna()
    if len(a_clean) < 3 or len(b_clean) < 3:
        return np.nan
    return float(stats.ttest_ind(a_clean, b_clean, equal_var=False, nan_policy="omit").pvalue)


def _status_from_pvalue(
    p_value: float,
    effect: float,
    alpha: float,
    sample_size: int,
    *,
    positive_effect_passes: bool = True,
) -> tuple[str, str]:
    if sample_size < MIN_REQUIRED_SAMPLE:
        return "insufficient_sample", "sample_size_below_minimum"
    if importlib.util.find_spec("scipy") is None:
        return "skipped_missing_dependency", "scipy_missing"
    if not np.isfinite(p_value):
        return "inconclusive", "p_value_unavailable"
    effect_ok = effect > 0.0 if positive_effect_passes else effect < 0.0
    return (
        ("passed", "statistically_significant")
        if p_value <= alpha and effect_ok
        else ("failed", "not_statistically_significant_or_wrong_direction")
    )


def _status_from_threshold(
    metric: float,
    comparison: float,
    sample_size: int,
    reason: str,
    *,
    greater_is_better: bool = True,
) -> tuple[str, str]:
    if sample_size < MIN_REQUIRED_SAMPLE:
        return "insufficient_sample", "sample_size_below_minimum"
    if not np.isfinite(metric) or not np.isfinite(comparison):
        return "inconclusive", "metric_unavailable"
    passed = metric > comparison if greater_is_better else metric < comparison
    return ("passed" if passed else "failed", reason)


def _anova_pvalue(groups: list[pd.Series]) -> float:
    if importlib.util.find_spec("scipy") is None:
        return np.nan
    from scipy import stats

    cleaned = [group.replace([np.inf, -np.inf], np.nan).dropna() for group in groups]
    cleaned = [group for group in cleaned if len(group) >= 3]
    if len(cleaned) < 2:
        return np.nan
    return float(stats.f_oneway(*cleaned).pvalue)


def _random_baseline_percentile(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[float, float]:
    returns = df.get("ret_1", pd.Series(index=df.index, dtype=float)).fillna(0.0)
    position = df.get("position", pd.Series(0.0, index=df.index)).fillna(0.0)
    exposure = float(position.abs().mean())
    strategy_sharpe = sharpe(
        df.get("strategy_ret", pd.Series(0.0, index=df.index)), cfg.bars_per_year
    )
    rng = np.random.default_rng(int(cfg.random_seed))
    samples = []
    dds = []
    for _ in range(100):
        random_position = pd.Series(
            rng.choice([-exposure, 0.0, exposure], size=len(returns), p=[0.25, 0.5, 0.25]),
            index=returns.index,
        )
        random_ret = random_position.shift(1).fillna(0.0) * returns
        equity = (1.0 + random_ret).cumprod()
        samples.append(sharpe(random_ret, cfg.bars_per_year))
        dds.append(max_drawdown(equity))
    sharpe_percentile = float(np.mean(np.asarray(samples) <= strategy_sharpe))
    dd_percentile = float(
        np.mean(np.asarray(dds) <= max_drawdown(df.get("strategy_equity", pd.Series(1.0))))
    )
    return sharpe_percentile, dd_percentile


def run_hypothesis_validation(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    forward = _forward_return(df, 20)
    bull = forward[df.get(STATE_BULL, pd.Series(0.0, index=df.index)) >= 0.5]
    bear = forward[df.get(STATE_BEAR, pd.Series(0.0, index=df.index)) >= 0.5]
    bull_vol = df.get("ret_1", pd.Series(index=df.index, dtype=float)).abs()[
        df.get(STATE_BULL, pd.Series(0.0, index=df.index)) >= 0.5
    ]
    bear_vol = df.get("ret_1", pd.Series(index=df.index, dtype=float)).abs()[
        df.get(STATE_BEAR, pd.Series(0.0, index=df.index)) >= 0.5
    ]
    bull_mean = float(bull.mean()) if len(bull.dropna()) else np.nan
    bear_mean = float(bear.mean()) if len(bear.dropna()) else np.nan
    spread = bull_mean - bear_mean if np.isfinite(bull_mean) and np.isfinite(bear_mean) else np.nan
    spread_std = float(pd.concat([bull, bear]).std(ddof=0))
    return_sample = int(len(pd.concat([bull, bear]).dropna()))
    return_p = _ttest_pvalue(bull, bear)
    return_status, return_reason = _status_from_pvalue(
        return_p,
        spread,
        cfg.hypothesis_alpha,
        return_sample,
    )
    vol_sample = int(len(pd.concat([bull_vol, bear_vol]).dropna()))
    vol_effect = float(bear_vol.mean() - bull_vol.mean())
    vol_p = _anova_pvalue([bull_vol, bear_vol])
    vol_status, vol_reason = _status_from_pvalue(
        vol_p,
        vol_effect,
        cfg.hypothesis_alpha,
        vol_sample,
    )
    sharpe_percentile, dd_percentile = _random_baseline_percentile(df, cfg)
    market_r2 = float(df.get("market_model_r2", pd.Series(np.nan, index=df.index)).mean())
    hidden = df.get("hidden_state", pd.Series(dtype=object))
    persistence = float(hidden.eq(hidden.shift(1)).mean())
    n_states = max(int(hidden.dropna().nunique()), 1)
    random_persistence = 1.0 / n_states
    persistence_status, persistence_reason = _status_from_threshold(
        persistence,
        random_persistence,
        int(hidden.notna().sum()),
        "deterministic_persistence_comparison",
    )
    sharpe_status, sharpe_reason = _status_from_threshold(
        sharpe_percentile,
        0.50,
        int(len(df)),
        "random_baseline_percentile_comparison",
    )
    dd_status, dd_reason = _status_from_threshold(
        dd_percentile,
        0.50,
        int(len(df)),
        "random_baseline_drawdown_percentile_comparison",
    )
    beta_status, beta_reason = _status_from_threshold(
        market_r2,
        0.70,
        int(df.get("market_model_r2", pd.Series(dtype=float)).notna().sum()),
        "market_beta_r2_threshold",
        greater_is_better=False,
    )
    rows = [
        {
            "test_name": "bull_vs_bear_forward_return",
            "hypothesis": "Bull-state forward returns exceed Bear-state forward returns after costs.",
            "status": return_status,
            "metric_name": "state_forward_return_tstat",
            "metric_value": spread / spread_std if spread_std and spread_std > 0 else np.nan,
            "comparison_value": 0.0,
            "p_value": return_p,
            "alpha": cfg.hypothesis_alpha,
            "effect_size": spread,
            "passed": return_status == "passed",
            "sample_size": return_sample,
            "min_required_sample": MIN_REQUIRED_SAMPLE,
            "window_id": "all",
            "reason": return_reason,
            "notes": "ex_post_validation_forward_returns_not_hmm_input",
        },
        {
            "test_name": "bear_volatility_exceeds_bull",
            "hypothesis": "Bear-state volatility exceeds Bull-state volatility.",
            "status": vol_status,
            "metric_name": "state_volatility_anova_pvalue",
            "metric_value": vol_effect,
            "comparison_value": 0.0,
            "p_value": vol_p,
            "alpha": cfg.hypothesis_alpha,
            "effect_size": vol_effect,
            "passed": vol_status == "passed",
            "sample_size": vol_sample,
            "min_required_sample": MIN_REQUIRED_SAMPLE,
            "window_id": "all",
            "reason": vol_reason,
            "notes": "absolute_return_group_test",
        },
        {
            "test_name": "state_persistence",
            "hypothesis": "State persistence is greater than random transition baseline.",
            "status": persistence_status,
            "metric_name": "transition_persistence_pvalue",
            "metric_value": persistence,
            "comparison_value": random_persistence,
            "p_value": np.nan,
            "alpha": cfg.hypothesis_alpha,
            "effect_size": persistence - random_persistence,
            "passed": persistence_status == "passed",
            "sample_size": int(hidden.notna().sum()),
            "min_required_sample": MIN_REQUIRED_SAMPLE,
            "window_id": "all",
            "reason": persistence_reason,
            "notes": "descriptive_without_closed_form_pvalue",
        },
        {
            "test_name": "random_timing_baseline",
            "hypothesis": "HMM strategy Sharpe exceeds random timing baseline.",
            "status": sharpe_status,
            "metric_name": "random_baseline_percentile",
            "metric_value": sharpe_percentile,
            "comparison_value": 0.50,
            "p_value": np.nan,
            "alpha": cfg.hypothesis_alpha,
            "effect_size": sharpe_percentile - 0.50,
            "passed": sharpe_status == "passed",
            "sample_size": int(len(df)),
            "min_required_sample": MIN_REQUIRED_SAMPLE,
            "window_id": "all",
            "reason": sharpe_reason,
            "notes": "deterministic_random_timing_simulation",
        },
        {
            "test_name": "drawdown_vs_random_timing",
            "hypothesis": "HMM max drawdown is lower than matched random timing baseline.",
            "status": dd_status,
            "metric_name": "random_baseline_drawdown_percentile",
            "metric_value": dd_percentile,
            "comparison_value": 0.50,
            "p_value": np.nan,
            "alpha": cfg.hypothesis_alpha,
            "effect_size": dd_percentile - 0.50,
            "passed": dd_status == "passed",
            "sample_size": int(len(df)),
            "min_required_sample": MIN_REQUIRED_SAMPLE,
            "window_id": "all",
            "reason": dd_reason,
            "notes": "drawdowns_are_negative_so_greater_is_less_severe",
        },
        {
            "test_name": "market_beta_explanation",
            "hypothesis": "HMM return is not purely explained by market beta.",
            "status": beta_status,
            "metric_name": "market_model_r2",
            "metric_value": market_r2,
            "comparison_value": 0.70,
            "p_value": np.nan,
            "alpha": cfg.hypothesis_alpha,
            "effect_size": 0.70 - market_r2 if np.isfinite(market_r2) else np.nan,
            "passed": beta_status == "passed",
            "sample_size": int(df.get("market_model_r2", pd.Series(dtype=float)).notna().sum()),
            "min_required_sample": MIN_REQUIRED_SAMPLE,
            "window_id": "all",
            "reason": beta_reason,
            "notes": "high_r2_means_signal_may_be_market_beta",
        },
    ]
    return pd.DataFrame(rows)


def statistical_validity_report(
    df: pd.DataFrame,
    hypothesis: pd.DataFrame,
    warnings: pd.DataFrame,
) -> pd.DataFrame:
    pvalues = hypothesis["p_value"].replace([np.inf, -np.inf], np.nan).dropna()
    return pd.DataFrame(
        [
            {
                "stationarity_risk": float(
                    df.get("stationarity_risk_score", pd.Series(dtype=float)).mean()
                ),
                "parameter_instability": float(
                    df.get("parameter_instability_z", pd.Series(dtype=float)).mean()
                ),
                "random_baseline_percentile": float(
                    hypothesis.loc[
                        hypothesis["metric_name"].eq("random_baseline_percentile"),
                        "metric_value",
                    ].mean()
                ),
                "p_value_availability": float(len(pvalues) / max(len(hypothesis), 1)),
                "validation_warning_count": int(0 if warnings is None else len(warnings)),
            }
        ]
    )
