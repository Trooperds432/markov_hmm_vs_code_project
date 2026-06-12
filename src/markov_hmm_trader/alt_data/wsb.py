from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig


def build_wsb_attention_features(
    base: pd.DataFrame, mentions: pd.DataFrame, cfg: PipelineConfig
) -> pd.DataFrame:
    features = pd.DataFrame(index=base.index)
    if mentions.empty:
        daily = pd.DataFrame(index=base.index)
    else:
        local = mentions.copy()
        local["available_date"] = local["date"] + pd.to_timedelta(
            int(cfg.wsb_publication_lag_days), unit="D"
        )
        numeric_defaults = {
            "mention_count": 0,
            "title_mentions": 0,
            "body_mentions": 0,
            "cashtag_mentions": 0,
            "bare_ticker_mentions": 0,
            "post_count": 0,
            "comment_count": 0,
            "score_sum": 0,
        }
        for column, default in numeric_defaults.items():
            if column not in local:
                local[column] = default
        daily = (
            local.groupby(["available_date", "ticker"], as_index=False)[list(numeric_defaults)]
            .sum()
            .sort_values("available_date")
        )
    ticker = str(cfg.ticker).upper()
    ticker_daily = daily[
        daily.get("ticker", pd.Series(dtype=str)).astype(str).str.upper().eq(ticker)
    ]
    ticker_daily = (
        ticker_daily.set_index("available_date") if not ticker_daily.empty else pd.DataFrame()
    )
    mentions_series = ticker_daily.get("mention_count", pd.Series(dtype=float)).reindex(
        base.index, method="ffill"
    )
    features["wsb_mentions"] = mentions_series.fillna(0.0)
    for window in cfg.wsb_z_windows:
        mean = (
            features["wsb_mentions"]
            .rolling(int(window), min_periods=max(3, int(window) // 3))
            .mean()
        )
        std = (
            features["wsb_mentions"]
            .rolling(int(window), min_periods=max(3, int(window) // 3))
            .std(ddof=0)
        )
        features[f"wsb_mentions_z_{int(window)}d"] = (
            features["wsb_mentions"] - mean
        ) / std.replace(0.0, np.nan)
    features["wsb_mention_change_1d"] = features["wsb_mentions"].diff()
    features["wsb_mention_change_5d"] = features["wsb_mentions"].diff(5)
    if not daily.empty:
        rank = daily.pivot(index="available_date", columns="ticker", values="mention_count").rank(
            axis=1, pct=True
        )
        features["wsb_rank_percentile"] = rank.get(ticker, pd.Series(dtype=float)).reindex(
            base.index, method="ffill"
        )
    else:
        features["wsb_rank_percentile"] = 0.5
    denominator = features["wsb_mentions"].replace(0.0, np.nan)
    for source, target in {
        "title_mentions": "wsb_title_mentions_share",
        "body_mentions": "wsb_body_mentions_share",
        "cashtag_mentions": "wsb_cashtag_mentions_share",
        "bare_ticker_mentions": "wsb_bare_ticker_mentions_share",
    }.items():
        source_series = ticker_daily.get(source, pd.Series(dtype=float)).reindex(
            base.index, method="ffill"
        )
        features[target] = source_series / denominator
    features["wsb_attention_volatility"] = (
        features["wsb_mention_change_1d"]
        .rolling(int(cfg.wsb_lookback_days), min_periods=5)
        .std(ddof=0)
    )
    baseline = features["wsb_mentions"].rolling(int(cfg.wsb_lookback_days), min_periods=5).mean()
    features["wsb_attention_shock"] = features["wsb_mentions"] / baseline.replace(0.0, np.nan)
    features["wsb_data_age_days"] = np.where(features["wsb_mentions"].notna(), 0, np.nan)
    z20 = features.get("wsb_mentions_z_20d", pd.Series(np.nan, index=features.index))
    features["wsb_meme_risk_score"] = pd.concat(
        [
            z20.clip(lower=0.0),
            features["wsb_attention_volatility"].rank(pct=True),
            features["wsb_cashtag_mentions_share"],
            features["wsb_title_mentions_share"],
            features["wsb_rank_percentile"],
        ],
        axis=1,
    ).mean(axis=1)
    features["wsb_attention_risk_flag"] = z20.gt(float(cfg.wsb_high_attention_z))
    features["wsb_extreme_attention_flag"] = z20.gt(float(cfg.wsb_extreme_attention_z))
    return features
