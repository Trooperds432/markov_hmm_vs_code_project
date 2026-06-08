from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from .backtest import walk_forward_backtest
from .config import PipelineConfig, apply_profile
from .data import load_market_data
from .features import make_features
from .metrics import add_backtest_metric_columns, make_report


def generate_nested_windows(
    index: pd.DatetimeIndex,
    train_years: int,
    validation_years: int,
    test_years: int,
    step_years: int,
) -> pd.DataFrame:
    if index.empty:
        return pd.DataFrame()

    rows = []
    start = index.min()
    final_date = index.max()
    window_id = 1
    while True:
        train_start = start
        train_end = train_start + pd.DateOffset(years=train_years)
        validation_end = train_end + pd.DateOffset(years=validation_years)
        test_end = validation_end + pd.DateOffset(years=test_years)
        if test_end > final_date:
            break
        rows.append(
            {
                "outer_window_id": window_id,
                "train_start": train_start,
                "train_end": train_end,
                "validation_start": train_end,
                "validation_end": validation_end,
                "test_start": validation_end,
                "test_end": test_end,
            }
        )
        window_id += 1
        start = start + pd.DateOffset(years=step_years)
    return pd.DataFrame(rows)


def score_stats(stats: pd.Series, objective: str) -> float:
    sharpe_value = float(stats.get("strategy_sharpe", np.nan))
    cagr_value = float(stats.get("strategy_cagr", np.nan))
    max_drawdown_value = abs(float(stats.get("strategy_max_drawdown", np.nan)))
    calmar_value = cagr_value / max_drawdown_value if max_drawdown_value else np.nan
    if objective == "sharpe_first":
        return sharpe_value
    if objective == "calmar_first":
        return calmar_value
    if objective == "aggressive_growth":
        return cagr_value + 0.25 * sharpe_value - 0.25 * max_drawdown_value
    if objective == "low_drawdown":
        return sharpe_value + calmar_value - max_drawdown_value
    return sharpe_value + 0.5 * cagr_value - 0.5 * max_drawdown_value


def candidate_configs(cfg: PipelineConfig) -> list[PipelineConfig]:
    profiles = ["balanced", "aggressive_growth", "low_drawdown"]
    candidates = [apply_profile(cfg, profile) for profile in profiles]
    return candidates[: max(int(cfg.nested_trials_per_window), 1)]


def _run_slice(features: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, pd.Series]:
    result = walk_forward_backtest(features, replace(cfg, enable_diagnostics=False))
    backtest = add_backtest_metric_columns(result.backtest, cfg)
    return backtest, make_report(backtest, cfg)


def run_nested_optimisation(cfg: PipelineConfig, output_dir) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = load_market_data(cfg)
    features = make_features(raw, cfg)
    windows = generate_nested_windows(
        features.index,
        cfg.outer_train_years,
        cfg.outer_validation_years,
        cfg.outer_test_years,
        cfg.outer_step_years,
    )

    window_rows = []
    candidate_rows = []
    oos_curves = []

    for _, window in windows.iterrows():
        outer_id = int(window["outer_window_id"])
        train_val = features.loc[window["train_start"] : window["validation_end"]]
        validation = features.loc[window["validation_start"] : window["validation_end"]]
        test = features.loc[window["test_start"] : window["test_end"]]
        train_val_test = features.loc[window["train_start"] : window["test_end"]]
        best_cfg = None
        best_score = -np.inf

        for candidate_id, candidate_cfg in enumerate(candidate_configs(cfg), start=1):
            try:
                backtest, stats = _run_slice(train_val, candidate_cfg)
                validation_stats = make_report(
                    backtest.reindex(validation.index).dropna(), candidate_cfg
                )
                validation_score = score_stats(validation_stats, cfg.nested_objective)
            except Exception as err:
                validation_score = -np.inf
                stats = pd.Series(dtype=float)
                candidate_rows.append(
                    {
                        "outer_window_id": outer_id,
                        "candidate_id": candidate_id,
                        "error": type(err).__name__,
                    }
                )
                continue

            row = stats.to_dict()
            row.update(
                {
                    "outer_window_id": outer_id,
                    "candidate_id": candidate_id,
                    "validation_score": validation_score,
                    "selected_config": json.dumps(asdict(candidate_cfg), default=str),
                }
            )
            candidate_rows.append(row)
            if validation_score > best_score:
                best_score = validation_score
                best_cfg = candidate_cfg

        if best_cfg is None:
            continue

        try:
            full_backtest, _ = _run_slice(train_val_test, best_cfg)
            test_backtest = full_backtest.reindex(test.index).dropna()
            test_stats = make_report(test_backtest, best_cfg)
            oos = test_backtest[["strategy_equity", "strategy_ret"]].copy()
            oos["outer_window_id"] = outer_id
            oos_curves.append(oos)
            window_rows.append(
                {
                    "outer_window_id": outer_id,
                    "train_start": window["train_start"],
                    "train_end": window["train_end"],
                    "validation_start": window["validation_start"],
                    "validation_end": window["validation_end"],
                    "test_start": window["test_start"],
                    "test_end": window["test_end"],
                    "selected_config": json.dumps(asdict(best_cfg), default=str),
                    "validation_score": best_score,
                    "test_total_return": test_stats.get("strategy_total_return", np.nan),
                    "test_cagr": test_stats.get("strategy_cagr", np.nan),
                    "test_max_drawdown": test_stats.get("strategy_max_drawdown", np.nan),
                    "test_sharpe": test_stats.get("strategy_sharpe", np.nan),
                    "test_turnover": test_stats.get("turnover", np.nan),
                    "test_win_rate": test_stats.get("win_rate", np.nan),
                    "test_profit_factor": test_stats.get("profit_factor", np.nan),
                }
            )
        except Exception as err:
            window_rows.append({"outer_window_id": outer_id, "error": type(err).__name__})

    outputs = {
        "nested_window_results": pd.DataFrame(window_rows),
        "nested_candidate_results": pd.DataFrame(candidate_rows),
        "nested_oos_equity_curve": pd.concat(oos_curves) if oos_curves else pd.DataFrame(),
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=name != "nested_oos_equity_curve")
    return outputs
