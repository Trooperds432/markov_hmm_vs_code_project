from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .leakage import FeatureLeakageGuard


@dataclass(frozen=True)
class FeatureQualityResult:
    frame: pd.DataFrame
    hmm_features: list[str]
    quality_report: pd.DataFrame
    drop_report: pd.DataFrame


def _effective_frame(df: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    """Ignore normal indicator warmup rows when measuring feature quality."""
    available = [col for col in requested if col in df]
    if not available:
        return df
    valid_any = df[available].replace([np.inf, -np.inf], np.nan).notna().any(axis=1)
    if not valid_any.any():
        return df
    return df.loc[valid_any.idxmax() :]


def _safe_stat(series: pd.Series, fn: str) -> float:
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return np.nan
    if fn == "std":
        return float(clean.std(ddof=0))
    if fn == "mean":
        return float(clean.mean())
    if fn == "min":
        return float(clean.min())
    if fn == "max":
        return float(clean.max())
    if fn == "p01":
        return float(clean.quantile(0.01))
    if fn == "p99":
        return float(clean.quantile(0.99))
    raise ValueError(fn)


def _correlation_details(frame: pd.DataFrame, features: list[str]) -> dict[str, tuple[str, float]]:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    if numeric.shape[1] < 2:
        return {feature: ("", np.nan) for feature in features}
    corr = numeric.corr().abs()
    details: dict[str, tuple[str, float]] = {}
    for feature in features:
        peers = corr[feature].drop(labels=[feature], errors="ignore").dropna()
        if peers.empty:
            details[feature] = ("", np.nan)
        else:
            peer = str(peers.idxmax())
            details[feature] = (peer, float(peers.loc[peer]))
    return details


def _source_map(metadata: pd.DataFrame) -> dict[str, str]:
    if metadata is None or metadata.empty or "feature" not in metadata:
        return {}
    group_col = "feature_group" if "feature_group" in metadata else "source_block"
    return {
        str(row.feature): str(getattr(row, group_col, "unknown"))
        for row in metadata.itertuples(index=False)
    }


def apply_feature_quality(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    requested_hmm_features: list[str],
    metadata: pd.DataFrame | None = None,
) -> FeatureQualityResult:
    frame = df.copy()
    guard = FeatureLeakageGuard()
    effective = _effective_frame(frame, requested_hmm_features)
    source_by_feature = _source_map(metadata if metadata is not None else pd.DataFrame())
    corr_details = _correlation_details(
        effective,
        [feature for feature in requested_hmm_features if feature in effective],
    )
    rows: list[dict[str, object]] = []
    drop_rows: list[dict[str, object]] = []
    kept: list[str] = []
    dropped_features: set[str] = set()

    for feature in requested_hmm_features:
        source = source_by_feature.get(feature, "legacy_base")
        if feature not in frame:
            drop_reason = "missing_feature"
            series = pd.Series(index=effective.index, dtype=float)
            inf_count = 0
        else:
            series = effective[feature]
            inf_count = int(np.isinf(pd.to_numeric(series, errors="coerce")).sum())
            drop_reason = ""

        clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing_ratio = float(clean.isna().mean()) if len(clean) else 1.0
        zero_variance = bool(clean.dropna().nunique() <= 1)
        leakage = guard.check_no_future_columns([feature])
        safe_for_hmm = leakage.status == "pass"
        winsorized = False

        if not safe_for_hmm:
            drop_reason = "validation_only"
        elif feature not in frame:
            drop_reason = "missing_feature"
        elif clean.notna().sum() == 0:
            drop_reason = "all_nan"
        elif inf_count and clean.notna().sum() == 0:
            drop_reason = "unrepairable_infinite"
        elif cfg.drop_high_missing_features and missing_ratio > cfg.max_feature_missing_ratio:
            drop_reason = "high_missing_ratio"
        elif cfg.drop_zero_variance_features and zero_variance:
            drop_reason = "zero_variance"

        peer, corr_value = corr_details.get(feature, ("", np.nan))
        if (
            not drop_reason
            and cfg.feature_corr_drop_threshold is not None
            and np.isfinite(corr_value)
            and corr_value >= cfg.feature_corr_drop_threshold
        ):
            drop_reason = "high_correlation"

        if not drop_reason and feature in frame:
            low, high = cfg.feature_winsorize_limits
            if 0.0 <= low < high <= 1.0:
                lower = clean.quantile(float(low))
                upper = clean.quantile(float(high))
                if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                    frame[feature] = pd.to_numeric(frame[feature], errors="coerce").clip(
                        lower=lower,
                        upper=upper,
                    )
                    winsorized = True
            kept.append(feature)
        else:
            dropped_features.add(feature)
            drop_rows.append(
                {
                    "feature": feature,
                    "source_block": source,
                    "drop_reason": drop_reason,
                    "missing_ratio": missing_ratio,
                    "used_in_hmm_requested": True,
                }
            )

        valid = clean.dropna()
        quality_status = "dropped" if drop_reason else "used"
        if (
            not drop_reason
            and np.isfinite(corr_value)
            and corr_value >= cfg.feature_corr_warn_threshold
        ):
            quality_status = "warn_high_correlation"
        elif (
            not drop_reason
            and np.isfinite(_safe_stat(clean, "std"))
            and _safe_stat(clean, "std") < 1e-12
        ):
            quality_status = "warn_low_variance"
        rows.append(
            {
                "feature": feature,
                "source_block": source,
                "used_in_hmm": feature in kept,
                "missing_count": int(clean.isna().sum()),
                "missing_ratio": missing_ratio,
                "inf_count": inf_count,
                "zero_variance": zero_variance,
                "std": _safe_stat(clean, "std"),
                "mean": _safe_stat(clean, "mean"),
                "min": _safe_stat(clean, "min"),
                "max": _safe_stat(clean, "max"),
                "p01": _safe_stat(clean, "p01"),
                "p99": _safe_stat(clean, "p99"),
                "winsorized": winsorized,
                "dropped": feature in dropped_features,
                "drop_reason": drop_reason,
                "correlation_cluster": peer if np.isfinite(corr_value) else "",
                "highest_abs_corr_feature": peer,
                "highest_abs_corr_value": corr_value,
                "first_valid_date": (
                    "" if valid.empty else pd.Timestamp(valid.index.min()).date().isoformat()
                ),
                "last_valid_date": (
                    "" if valid.empty else pd.Timestamp(valid.index.max()).date().isoformat()
                ),
                "quality_status": quality_status,
                "safe_for_hmm": safe_for_hmm,
                "validation_only": not safe_for_hmm,
            }
        )

    if not kept:
        raise ValueError("Feature quality dropped every configured HMM observation feature.")

    report = pd.DataFrame(rows)
    drop_report = pd.DataFrame(
        drop_rows,
        columns=[
            "feature",
            "source_block",
            "drop_reason",
            "missing_ratio",
            "used_in_hmm_requested",
        ],
    )
    frame.attrs["hmm_feature_columns"] = kept
    frame.attrs["feature_quality_report"] = report.to_dict("records")
    frame.attrs["feature_drop_report"] = drop_report.to_dict("records")
    return FeatureQualityResult(frame, kept, report, drop_report)


def hmm_feature_columns(df: pd.DataFrame, cfg: PipelineConfig, requested: list[str]) -> list[str]:
    attrs = df.attrs.get("hmm_feature_columns")
    if isinstance(attrs, list) and attrs:
        return [str(item) for item in attrs]
    return requested


def feature_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    records = df.attrs.get("feature_quality_report", [])
    return pd.DataFrame(records)


def feature_drop_report(df: pd.DataFrame) -> pd.DataFrame:
    records = df.attrs.get("feature_drop_report", [])
    return pd.DataFrame(records)
