from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .metrics import cagr, max_drawdown, sharpe
from .states import STATE_BEAR, STATE_BULL, STATE_ORDER, STATE_SIDE


def label_observable_states(df: pd.DataFrame, cfg: PipelineConfig) -> pd.Series:
    rolling_return = df["ret_1"].rolling(cfg.observable_label_window).sum()
    labels = pd.Series(STATE_SIDE, index=df.index, dtype="object")
    labels[rolling_return >= cfg.observable_bull_threshold] = STATE_BULL
    labels[rolling_return <= cfg.observable_bear_threshold] = STATE_BEAR
    return labels


def transition_matrix(labels: pd.Series) -> pd.DataFrame:
    counts = pd.DataFrame(0.0, index=STATE_ORDER, columns=STATE_ORDER)
    previous = labels.shift(1)
    for from_state, to_state in zip(previous.iloc[1:], labels.iloc[1:], strict=False):
        if from_state in STATE_ORDER and to_state in STATE_ORDER:
            counts.loc[from_state, to_state] += 1.0
    totals = counts.sum(axis=1).replace(0.0, np.nan)
    matrix = counts.div(totals, axis=0).fillna(1.0 / len(STATE_ORDER))
    return matrix


def stationary_distribution(matrix: pd.DataFrame) -> pd.Series:
    values, vectors = np.linalg.eig(matrix.to_numpy(dtype=float).T)
    index = int(np.argmin(np.abs(values - 1.0)))
    stationary = np.real(vectors[:, index])
    stationary = np.maximum(stationary, 0.0)
    if stationary.sum() == 0.0:
        stationary = np.ones(len(STATE_ORDER)) / len(STATE_ORDER)
    else:
        stationary = stationary / stationary.sum()
    return pd.Series(stationary, index=STATE_ORDER)


def n_step_forecast(matrix: pd.DataFrame, current_state: str, steps: int) -> pd.Series:
    start = np.zeros(len(STATE_ORDER))
    start[STATE_ORDER.index(current_state)] = 1.0
    probabilities = start @ np.linalg.matrix_power(matrix.to_numpy(dtype=float), int(steps))
    return pd.Series(probabilities, index=STATE_ORDER)


def run_observable_markov(df: pd.DataFrame, cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    labels = label_observable_states(df, cfg)
    matrix = transition_matrix(labels.dropna())
    stationary = stationary_distribution(matrix)

    transition_rows = []
    for from_state in STATE_ORDER:
        for to_state in STATE_ORDER:
            transition_rows.append(
                {
                    "date": df.index[-1].date().isoformat(),
                    "from_state": from_state,
                    "to_state": to_state,
                    "probability": float(matrix.loc[from_state, to_state]),
                }
            )

    forecast_rows = []
    for date, state in labels.dropna().items():
        for steps in cfg.observable_forecast_steps:
            forecast = n_step_forecast(matrix, state, int(steps))
            signal = float(forecast[STATE_BULL] - forecast[STATE_BEAR])
            forecast_rows.append(
                {
                    "date": pd.Timestamp(date).date().isoformat(),
                    "forecast_steps": int(steps),
                    "p_bear": float(forecast[STATE_BEAR]),
                    "p_sideways": float(forecast[STATE_SIDE]),
                    "p_bull": float(forecast[STATE_BULL]),
                    "signal": signal,
                    "stationary_bear": float(stationary[STATE_BEAR]),
                    "stationary_sideways": float(stationary[STATE_SIDE]),
                    "stationary_bull": float(stationary[STATE_BULL]),
                }
            )

    signal = labels.map({STATE_BULL: 1.0, STATE_SIDE: 0.0, STATE_BEAR: 0.0}).fillna(0.0)
    strategy_return = signal.shift(1).fillna(0.0) * df["ret_1"].fillna(0.0)
    equity = (1.0 + strategy_return).cumprod()
    equity_curve = pd.DataFrame(
        {
            "date": df.index.date.astype(str),
            "position": signal.values,
            "strategy_return": strategy_return.values,
            "equity": equity.values,
            "drawdown": (equity / equity.cummax() - 1.0).values,
        },
        index=df.index,
    )
    stats = pd.DataFrame(
        [
            {
                "model": "observable_markov",
                "cagr": cagr(equity, cfg.bars_per_year),
                "sharpe": sharpe(strategy_return, cfg.bars_per_year),
                "max_drawdown": max_drawdown(equity),
            }
        ]
    )
    return {
        "observable_markov_transition_matrix": pd.DataFrame(transition_rows),
        "observable_markov_forecasts": pd.DataFrame(forecast_rows),
        "observable_markov_equity_curve": equity_curve,
        "observable_markov_stats": stats,
    }
