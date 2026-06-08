from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig


def zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def percent_rank(series: pd.Series, window: int) -> pd.Series:
    def rank_last(x: np.ndarray) -> float:
        if len(x) == 0 or np.isnan(x[-1]):
            return np.nan
        return float(np.sum(x <= x[-1]) / len(x))

    return series.rolling(window).apply(rank_last, raw=True)


def feature_columns() -> list[str]:
    return [
        "z_ret_1",
        "z_ret_5",
        "z_ret_20",
        "z_rel_20",
        "z_ema_slope",
        "z_rv_20",
        "z_atr_rank",
        "z_volume",
        "z_range",
    ]


def make_features(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = np.log(out["Close"] / out["Close"].shift(1))
    out["ret_5"] = np.log(out["Close"] / out["Close"].shift(5))
    out["ret_20"] = np.log(out["Close"] / out["Close"].shift(20))

    out["bench_ret_1"] = np.log(out["Benchmark_Close"] / out["Benchmark_Close"].shift(1))
    out["rel_ret_1"] = out["ret_1"] - out["bench_ret_1"]
    out["rel_ret_20"] = out["ret_20"] - np.log(out["Benchmark_Close"] / out["Benchmark_Close"].shift(20))

    out["ema_fast"] = out["Close"].ewm(span=cfg.fast_ema, adjust=False).mean()
    out["ema_slow"] = out["Close"].ewm(span=cfg.slow_ema, adjust=False).mean()

    out["trend_long"] = (out["Close"] > out["ema_slow"]) & (out["ema_fast"] > out["ema_slow"])
    out["trend_short"] = (out["Close"] < out["ema_slow"]) & (out["ema_fast"] < out["ema_slow"])

    out["ema_slope"] = out["ema_fast"].pct_change(10)

    out["tr"] = true_range(out)
    out["atr"] = out["tr"].rolling(14).mean()
    out["atr_pct"] = out["atr"] / out["Close"]
    out["atr_rank"] = percent_rank(out["atr_pct"], 120)

    out["rv_20"] = out["ret_1"].rolling(20).std(ddof=0) * np.sqrt(cfg.bars_per_year)
    out["rv_realised"] = out["ret_1"].rolling(cfg.vol_lookback).std(ddof=0) * np.sqrt(cfg.bars_per_year)

    out["vol_z"] = zscore(np.log(out["Volume"].replace(0, np.nan)), 60)

    out["rs"] = out["Close"] / out["Benchmark_Close"]
    out["rs_fast"] = out["rs"].ewm(span=cfg.rs_fast, adjust=False).mean()
    out["rs_slow"] = out["rs"].ewm(span=cfg.rs_slow, adjust=False).mean()
    out["rs_long"] = out["rs_fast"] > out["rs_slow"]
    out["rs_short"] = out["rs_fast"] < out["rs_slow"]

    out["range_pct"] = (out["High"] - out["Low"]) / out["Close"]

    feature_map = {
        "z_ret_1": zscore(out["ret_1"], 80),
        "z_ret_5": zscore(out["ret_5"], 80),
        "z_ret_20": zscore(out["ret_20"], 120),
        "z_rel_20": zscore(out["rel_ret_20"], 120),
        "z_ema_slope": zscore(out["ema_slope"], 120),
        "z_rv_20": zscore(out["rv_20"], 120),
        "z_atr_rank": zscore(out["atr_rank"], 120),
        "z_volume": out["vol_z"],
        "z_range": zscore(out["range_pct"], 120),
    }

    for name, values in feature_map.items():
        out[name] = values.replace([np.inf, -np.inf], np.nan)

    out["alpha_score_long"] = 0
    out["alpha_score_long"] += out["trend_long"].astype(int)
    out["alpha_score_long"] += out["rs_long"].astype(int)
    out["alpha_score_long"] += (out["ret_20"] > 0).astype(int)
    out["alpha_score_long"] += (out["ema_slope"] > 0).astype(int)
    out["alpha_score_long"] += out["atr_rank"].between(0.08, 0.96).astype(int)

    out["alpha_score_short"] = 0
    out["alpha_score_short"] += out["trend_short"].astype(int)
    out["alpha_score_short"] += out["rs_short"].astype(int)
    out["alpha_score_short"] += (out["ret_20"] < 0).astype(int)
    out["alpha_score_short"] += (out["ema_slope"] < 0).astype(int)
    out["alpha_score_short"] += out["atr_rank"].between(0.08, 0.96).astype(int)

    return out.replace([np.inf, -np.inf], np.nan).dropna().copy()
