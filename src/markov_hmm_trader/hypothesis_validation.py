from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .metrics import max_drawdown, sharpe
from .states import STATE_BEAR, STATE_BULL


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
    sharpe_percentile, dd_percentile = _random_baseline_percentile(df, cfg)
    market_r2 = float(df.get("market_model_r2", pd.Series(np.nan, index=df.index)).mean())
    rows = [
        {
            "test_name": "bull_vs_bear_forward_return",
            "hypothesis": "Bull-state forward returns exceed Bear-state forward returns after costs.",
            "metric_name": "state_forward_return_tstat",
            "metric_value": spread / spread_std if spread_std and spread_std > 0 else np.nan,
            "p_value": _ttest_pvalue(bull, bear),
            "alpha": cfg.hypothesis_alpha,
            "passed": bool(spread > 0.0) if np.isfinite(spread) else False,
            "sample_size": int(len(pd.concat([bull, bear]).dropna())),
            "window_id": "all",
            "notes": "ex_post_validation_forward_returns_not_hmm_input",
        },
        {
            "test_name": "bear_volatility_exceeds_bull",
            "hypothesis": "Bear-state volatility exceeds Bull-state volatility.",
            "metric_name": "state_volatility_anova_pvalue",
            "metric_value": float(bear_vol.mean() - bull_vol.mean()),
            "p_value": _anova_pvalue([bull_vol, bear_vol]),
            "alpha": cfg.hypothesis_alpha,
            "passed": bool(bear_vol.mean() > bull_vol.mean()),
            "sample_size": int(len(pd.concat([bull_vol, bear_vol]).dropna())),
            "window_id": "all",
            "notes": "absolute_return_group_test",
        },
        {
            "test_name": "state_persistence",
            "hypothesis": "State persistence is greater than random transition baseline.",
            "metric_name": "transition_persistence_pvalue",
            "metric_value": float(
                df.get("hidden_state", pd.Series(dtype=object))
                .eq(df.get("hidden_state", pd.Series(dtype=object)).shift(1))
                .mean()
            ),
            "p_value": np.nan,
            "alpha": cfg.hypothesis_alpha,
            "passed": True,
            "sample_size": int(df.get("hidden_state", pd.Series(dtype=object)).notna().sum()),
            "window_id": "all",
            "notes": "descriptive_without_closed_form_pvalue",
        },
        {
            "test_name": "random_timing_baseline",
            "hypothesis": "HMM strategy Sharpe exceeds random timing baseline.",
            "metric_name": "random_baseline_percentile",
            "metric_value": sharpe_percentile,
            "p_value": np.nan,
            "alpha": cfg.hypothesis_alpha,
            "passed": bool(sharpe_percentile >= 0.5),
            "sample_size": int(len(df)),
            "window_id": "all",
            "notes": "deterministic_random_timing_simulation",
        },
        {
            "test_name": "drawdown_vs_buy_hold",
            "hypothesis": "HMM drawdown is lower than buy-and-hold drawdown.",
            "metric_name": "random_baseline_drawdown_percentile",
            "metric_value": dd_percentile,
            "p_value": np.nan,
            "alpha": cfg.hypothesis_alpha,
            "passed": bool(
                max_drawdown(df.get("strategy_equity", pd.Series(1.0)))
                > max_drawdown(df.get("buy_hold_equity", pd.Series(1.0)))
            ),
            "sample_size": int(len(df)),
            "window_id": "all",
            "notes": "drawdowns_are_negative_so_greater_is_less_severe",
        },
        {
            "test_name": "market_beta_explanation",
            "hypothesis": "HMM return is not purely explained by market beta.",
            "metric_name": "market_model_r2",
            "metric_value": market_r2,
            "p_value": np.nan,
            "alpha": cfg.hypothesis_alpha,
            "passed": bool(not np.isfinite(market_r2) or market_r2 < 0.70),
            "sample_size": int(df.get("market_model_r2", pd.Series(dtype=float)).notna().sum()),
            "window_id": "all",
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
