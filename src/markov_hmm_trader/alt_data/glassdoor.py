from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PipelineConfig

RATING_COLUMNS = ("overall_rating", "senior_mgmt", "culture_values", "work_life_balance")
CATEGORY_COLUMNS = {
    "recommend": "recommend",
    "ceo_approv": "ceo_approv",
    "business_outlook": "outlook",
}


def _cat_share(frame: pd.DataFrame, column: str, label: str) -> float:
    if column not in frame or frame.empty:
        return np.nan
    normalized = frame[column].astype(str).str.lower()
    positive = {"yes", "y", "positive", "approve", "approved", "1", "true"}
    negative = {"no", "n", "negative", "disapprove", "disapproved", "-1", "false"}
    if label == "positive":
        return float(normalized.isin(positive).mean())
    if label == "negative":
        return float(normalized.isin(negative).mean())
    return float((~normalized.isin(positive | negative)).mean())


def build_glassdoor_features(
    base: pd.DataFrame, reviews: pd.DataFrame, cfg: PipelineConfig
) -> pd.DataFrame:
    features = pd.DataFrame(index=base.index)
    columns = [
        "overall_rating_mean",
        "overall_rating_std",
        "senior_mgmt_mean",
        "senior_mgmt_std",
        "culture_values_mean",
        "culture_values_std",
        "work_life_balance_mean",
        "work_life_balance_std",
        "recommend_positive_share",
        "recommend_neutral_share",
        "recommend_negative_share",
        "ceo_approv_positive_share",
        "ceo_approv_neutral_share",
        "ceo_approv_negative_share",
        "outlook_positive_share",
        "outlook_neutral_share",
        "outlook_negative_share",
        "n_reviews",
        "n_reviews_last_90d",
        "n_reviews_last_365d",
        "days_since_last_review",
        "glassdoor_data_age_days",
    ]
    if reviews.empty:
        for column in columns:
            features[column] = np.nan
    else:
        local = reviews.copy()
        local["available_date"] = local["review_date"] + pd.to_timedelta(
            int(cfg.glassdoor_publication_lag_days), unit="D"
        )
        for date in features.index:
            as_of = pd.Timestamp(date).tz_localize(None)
            window_start = as_of - pd.Timedelta(days=int(cfg.glassdoor_lookback_days))
            trailing = local[
                local["available_date"].le(as_of) & local["review_date"].ge(window_start)
            ]
            for rating in RATING_COLUMNS:
                values = pd.to_numeric(trailing.get(rating), errors="coerce")
                features.loc[date, f"{rating}_mean"] = values.mean()
                features.loc[date, f"{rating}_std"] = values.std(ddof=0)
            for source, prefix in CATEGORY_COLUMNS.items():
                for label in ("positive", "neutral", "negative"):
                    features.loc[date, f"{prefix}_{label}_share"] = _cat_share(
                        trailing, source, label
                    )
            features.loc[date, "n_reviews"] = len(trailing)
            for window in cfg.glassdoor_recent_windows:
                recent = local[
                    local["available_date"].le(as_of)
                    & local["review_date"].ge(as_of - pd.Timedelta(days=int(window)))
                ]
                features.loc[date, f"n_reviews_last_{int(window)}d"] = len(recent)
            last_review = local[local["available_date"].le(as_of)]["review_date"].max()
            features.loc[date, "days_since_last_review"] = (
                (as_of - last_review).days if pd.notna(last_review) else np.nan
            )
            features.loc[date, "glassdoor_data_age_days"] = features.loc[
                date, "days_since_last_review"
            ]
    reviews_365 = features.get("n_reviews_last_365d", features["n_reviews"])
    features["glassdoor_review_intensity"] = features["n_reviews_last_90d"] / reviews_365.replace(
        0.0, np.nan
    )
    rating_score = (
        features[
            [
                "overall_rating_mean",
                "senior_mgmt_mean",
                "culture_values_mean",
                "work_life_balance_mean",
            ]
        ].mean(axis=1)
        / 5.0
    )
    sentiment_shares = features[
        ["recommend_positive_share", "outlook_positive_share", "ceo_approv_positive_share"]
    ].mean(axis=1)
    features["glassdoor_sentiment_score"] = pd.concat(
        [rating_score, sentiment_shares], axis=1
    ).mean(axis=1)
    features["glassdoor_stress_score"] = pd.concat(
        [
            1.0 - features["overall_rating_mean"] / 5.0,
            1.0 - features["senior_mgmt_mean"] / 5.0,
            features["outlook_negative_share"],
            features["ceo_approv_negative_share"],
            features["glassdoor_review_intensity"],
        ],
        axis=1,
    ).mean(axis=1)
    stress_window = max(int(cfg.glassdoor_stress_z_window), 3)
    stress_mean = features["glassdoor_stress_score"].rolling(stress_window, min_periods=3).mean()
    stress_std = (
        features["glassdoor_stress_score"].rolling(stress_window, min_periods=3).std(ddof=0)
    )
    features["glassdoor_stress_z"] = (
        features["glassdoor_stress_score"] - stress_mean
    ) / stress_std.replace(0.0, np.nan)
    return features
