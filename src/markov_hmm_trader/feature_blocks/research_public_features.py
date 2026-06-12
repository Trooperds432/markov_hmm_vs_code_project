from __future__ import annotations

import importlib.util
from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from ..warnings import StructuredWarning, warning_to_row
from .base import FeatureBlockResult, feature_metadata

RESEARCH_PUBLIC_GROUP = "research_public"


def _min_periods(window: int, cfg: PipelineConfig) -> int:
    return max(5, int(round(window * float(cfg.min_periods_ratio))))


def _zscore(series: pd.Series, window: int = 120) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 3)).mean()
    std = series.rolling(window, min_periods=max(5, window // 3)).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def _safe_log_return(df: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    if column not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    close = df[column].astype(float).ffill()
    return np.log(close / close.shift(periods)).replace([np.inf, -np.inf], np.nan)


def _rolling_beta(y: pd.Series, x: pd.Series, window: int, min_periods: int) -> pd.Series:
    cov = y.rolling(window, min_periods=min_periods).cov(x)
    var = x.rolling(window, min_periods=min_periods).var(ddof=0)
    return cov / var.replace(0.0, np.nan)


def _rolling_residual(y: pd.Series, x: pd.Series, beta: pd.Series) -> pd.Series:
    return y - beta * x


def _rolling_r2(y: pd.Series, residual: pd.Series, window: int, min_periods: int) -> pd.Series:
    residual_var = residual.rolling(window, min_periods=min_periods).var(ddof=0)
    y_var = y.rolling(window, min_periods=min_periods).var(ddof=0)
    return (1.0 - residual_var / y_var.replace(0.0, np.nan)).clip(lower=0.0, upper=1.0)


def _rolling_autocovariance(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    lagged = series.shift(1)
    mean = series.rolling(window, min_periods=min_periods).mean()
    lagged_mean = lagged.rolling(window, min_periods=min_periods).mean()
    return (
        ((series - mean) * (lagged - lagged_mean)).rolling(window, min_periods=min_periods).mean()
    )


def _rolling_hurst(log_price: pd.Series, window: int, min_periods: int) -> pd.Series:
    def hurst(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) < min_periods:
            return np.nan
        max_lag = min(20, max(3, len(values) // 2))
        lags = np.arange(2, max_lag + 1)
        tau = []
        for lag in lags:
            diff = values[lag:] - values[:-lag]
            tau.append(np.nanstd(diff))
        tau_arr = np.asarray(tau)
        valid = (tau_arr > 0.0) & np.isfinite(tau_arr)
        if valid.sum() < 3:
            return np.nan
        slope = np.polyfit(np.log(lags[valid]), np.log(tau_arr[valid]), 1)[0]
        return float(np.clip(slope, 0.0, 1.5))

    return log_price.rolling(window, min_periods=min_periods).apply(hurst, raw=True)


def _ewma_vol(ret: pd.Series, span: int, bars_per_year: int) -> pd.Series:
    # Shifted so the variance assigned to t only uses returns through t-1.
    variance = (
        ret.shift(1).pow(2).ewm(span=span, adjust=False, min_periods=max(5, span // 3)).mean()
    )
    return np.sqrt(variance.clip(lower=0.0)) * np.sqrt(bars_per_year)


def _tail_ratio(ret: pd.Series, window: int, min_periods: int) -> pd.Series:
    realised = ret.rolling(window, min_periods=min_periods).std(ddof=0)
    tail_hit = (ret.abs() > 2.0 * realised).astype(float)
    expected_normal_tail = 0.0455
    return tail_hit.rolling(window, min_periods=min_periods).mean() / expected_normal_tail


def _rolling_dw(residual: pd.Series, window: int, min_periods: int) -> pd.Series:
    def durbin_watson(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) < min_periods:
            return np.nan
        denom = np.sum(values**2)
        if denom <= 0.0:
            return np.nan
        return float(np.sum(np.diff(values) ** 2) / denom)

    return residual.rolling(window, min_periods=min_periods).apply(durbin_watson, raw=True)


def _rolling_jb_pvalue(residual: pd.Series, window: int, min_periods: int) -> pd.Series:
    if importlib.util.find_spec("scipy") is None:
        return pd.Series(np.nan, index=residual.index, dtype=float)
    from scipy.stats import jarque_bera

    def pvalue(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) < min_periods:
            return np.nan
        try:
            return float(jarque_bera(values).pvalue)
        except Exception:
            return np.nan

    return residual.rolling(window, min_periods=min_periods).apply(pvalue, raw=True)


def _rolling_adf_pvalue(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    if importlib.util.find_spec("statsmodels") is None:
        return pd.Series(np.nan, index=series.index, dtype=float)
    from statsmodels.tsa.stattools import adfuller

    def pvalue(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) < min_periods or np.nanstd(values) <= 0.0:
            return np.nan
        try:
            return float(adfuller(values, autolag=None, maxlag=min(5, len(values) // 4))[1])
        except Exception:
            return np.nan

    return series.rolling(window, min_periods=min_periods).apply(pvalue, raw=True)


def _rolling_half_life(spread: pd.Series, window: int, min_periods: int) -> pd.Series:
    def half_life(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if len(values) < min_periods:
            return np.nan
        lagged = values[:-1]
        delta = np.diff(values)
        denom = np.var(lagged)
        if denom <= 0.0:
            return np.nan
        beta = np.cov(delta, lagged, ddof=0)[0, 1] / denom
        if beta >= 0.0:
            return np.inf
        return float(np.clip(-np.log(2.0) / beta, 0.0, 10_000.0))

    return spread.rolling(window, min_periods=min_periods).apply(half_life, raw=True)


def _available_close_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "Close",
        "Benchmark_Close",
        "market_Close",
        "tech_Close",
        "semiconductors_Close",
        "growth_proxy_Close",
        "value_proxy_Close",
        "credit_proxy_Close",
    ]
    return [column for column in candidates if column in df]


def _warning(reason: str, detail: str, feature: str = "") -> dict[str, object]:
    return warning_to_row(
        StructuredWarning(
            code=reason,
            severity="warning",
            module="feature_blocks.research_public",
            message=f"Research public feature warning: {reason}.",
            details={"feature_group": RESEARCH_PUBLIC_GROUP, "feature": feature, "detail": detail},
        ),
        feature_group=RESEARCH_PUBLIC_GROUP,
        feature=feature,
        reason=reason,
        detail=detail,
    )


def _add_stationarity(
    features: pd.DataFrame,
    log_price: pd.Series,
    ret: pd.Series,
    cfg: PipelineConfig,
    warning_rows: list[dict[str, object]],
) -> None:
    if importlib.util.find_spec("statsmodels") is None:
        warning_rows.append(_warning("missing_optional_dependency", "statsmodels", "adf_pvalue"))
    instability_inputs = []
    for window in cfg.stationarity_windows:
        minp = _min_periods(window, cfg)
        features[f"adf_pvalue_{window}d"] = _rolling_adf_pvalue(log_price, window, minp)
        rolling_mean = ret.rolling(window, min_periods=minp).mean()
        rolling_vol = ret.rolling(window, min_periods=minp).std(ddof=0)
        autocov = _rolling_autocovariance(ret, window, minp)
        features[f"rolling_mean_instability_{window}d"] = rolling_mean.rolling(
            window, min_periods=minp
        ).std(ddof=0)
        features[f"rolling_vol_instability_{window}d"] = rolling_vol.rolling(
            window, min_periods=minp
        ).std(ddof=0)
        features[f"rolling_autocov_instability_{window}d"] = autocov.rolling(
            window, min_periods=minp
        ).std(ddof=0)
        if window == 120:
            features["hurst_exponent_120d"] = _rolling_hurst(log_price, window, minp)
        instability_inputs.extend(
            [
                features[f"adf_pvalue_{window}d"],
                _zscore(features[f"rolling_mean_instability_{window}d"].abs()),
                _zscore(features[f"rolling_vol_instability_{window}d"].abs()),
            ]
        )
    features["rolling_mean_instability"] = features.get(
        "rolling_mean_instability_60d", pd.Series(np.nan, index=features.index)
    )
    features["rolling_vol_instability"] = features.get(
        "rolling_vol_instability_60d", pd.Series(np.nan, index=features.index)
    )
    features["stationarity_risk_score"] = pd.concat(instability_inputs, axis=1).mean(axis=1)


def _add_autoregressive(
    features: pd.DataFrame,
    ret: pd.Series,
    cfg: PipelineConfig,
) -> None:
    for lag in (1, 2, 5):
        features[f"return_lag_{lag}"] = ret.shift(lag)
    for lag in cfg.acf_lags:
        features[f"acf_{lag}"] = ret.rolling(60, min_periods=_min_periods(60, cfg)).corr(
            ret.shift(lag)
        )
    features["pacf_1"] = features.get("acf_1", pd.Series(np.nan, index=features.index))
    lagged = ret.shift(1)
    beta = _rolling_beta(ret, lagged, 60, _min_periods(60, cfg))
    residual = ret - beta * lagged
    features["ar1_beta_60d"] = beta
    features["ar1_residual_zscore"] = _zscore(residual, 60)
    features["ar1_beta_instability"] = beta.rolling(60, min_periods=_min_periods(60, cfg)).std(
        ddof=0
    )
    features["rolling_ar1_instability"] = features["ar1_beta_instability"]


def _add_volatility_clustering(features: pd.DataFrame, ret: pd.Series, cfg: PipelineConfig) -> None:
    features["ewma_vol_20d"] = _ewma_vol(ret, 20, cfg.bars_per_year)
    features["ewma_vol_60d"] = _ewma_vol(ret, 60, cfg.bars_per_year)
    features["vol_ratio_20_60"] = features["ewma_vol_20d"] / features["ewma_vol_60d"].replace(
        0.0, np.nan
    )
    features["squared_return_lag_1"] = ret.shift(1).pow(2)
    features["abs_return_lag_1"] = ret.shift(1).abs()
    features["vol_of_vol_20d"] = (
        features["ewma_vol_20d"].rolling(20, min_periods=_min_periods(20, cfg)).std(ddof=0)
    )
    features["vol_shock_zscore"] = _zscore(features["ewma_vol_20d"], 60)
    features["tail_ratio_vs_normal"] = _tail_ratio(ret, 60, _min_periods(60, cfg))
    score_inputs = pd.DataFrame(
        {
            "vol_ratio": _zscore(features["vol_ratio_20_60"], 60),
            "vol_shock": features["vol_shock_zscore"],
            "vol_of_vol": _zscore(features["vol_of_vol_20d"], 60),
            "tail_ratio": _zscore(features["tail_ratio_vs_normal"], 60),
        },
        index=features.index,
    )
    features["vol_cluster_score"] = score_inputs.mean(axis=1)
    features["garch_sigma"] = np.nan


def _add_market_residual_and_correlation(
    features: pd.DataFrame,
    df: pd.DataFrame,
    ret: pd.Series,
    cfg: PipelineConfig,
) -> None:
    bench_ret = _safe_log_return(df, "Benchmark_Close", 1)
    market_ret = _safe_log_return(df, "market_Close", 1).fillna(bench_ret)
    qqq_ret = _safe_log_return(df, "tech_Close", 1).fillna(bench_ret)
    sector_ret = _safe_log_return(df, "semiconductors_Close", 1).fillna(qqq_ret)
    beta_window = int(cfg.rolling_beta_window)
    corr_window = int(cfg.rolling_corr_window)
    beta_minp = _min_periods(beta_window, cfg)
    corr_minp = _min_periods(corr_window, cfg)

    features["rolling_beta_spy_60d"] = _rolling_beta(ret, market_ret, beta_window, beta_minp)
    features["rolling_beta_qqq_60d"] = _rolling_beta(ret, qqq_ret, beta_window, beta_minp)
    features["rolling_beta_sector_60d"] = _rolling_beta(ret, sector_ret, beta_window, beta_minp)
    features["market_model_beta"] = _rolling_beta(ret, bench_ret, beta_window, beta_minp)
    features["market_model_alpha"] = (
        ret.rolling(beta_window, min_periods=beta_minp).mean()
        - features["market_model_beta"]
        * bench_ret.rolling(beta_window, min_periods=beta_minp).mean()
    )
    residual = _rolling_residual(ret, bench_ret, features["market_model_beta"])
    features["market_residual_return"] = residual
    features["market_residual_vol"] = residual.rolling(beta_window, min_periods=beta_minp).std(
        ddof=0
    )
    features["market_residual_zscore"] = _zscore(residual, beta_window)
    features["idiosyncratic_momentum_20d"] = residual.rolling(
        20, min_periods=_min_periods(20, cfg)
    ).sum()
    features["market_model_r2"] = _rolling_r2(ret, residual, beta_window, beta_minp)
    features["market_model_residual_skew"] = residual.rolling(
        beta_window, min_periods=beta_minp
    ).skew()
    features["market_model_residual_kurtosis"] = residual.rolling(
        beta_window, min_periods=beta_minp
    ).kurt()
    features["market_model_residual_vol"] = features["market_residual_vol"]
    features["durbin_watson_residuals"] = _rolling_dw(residual, beta_window, beta_minp)
    features["jarque_bera_pvalue"] = _rolling_jb_pvalue(residual, beta_window, beta_minp)
    tail_inputs = pd.DataFrame(
        {
            "residual_z": features["market_residual_zscore"].abs(),
            "kurtosis": _zscore(features["market_model_residual_kurtosis"].abs(), beta_window),
            "vol": _zscore(features["market_residual_vol"], beta_window),
        },
        index=features.index,
    )
    features["residual_tail_risk_score"] = tail_inputs.mean(axis=1)

    features["corr_spy_20d"] = ret.rolling(20, min_periods=_min_periods(20, cfg)).corr(market_ret)
    features["corr_spy_60d"] = ret.rolling(corr_window, min_periods=corr_minp).corr(market_ret)
    features["corr_qqq_60d"] = ret.rolling(corr_window, min_periods=corr_minp).corr(qqq_ret)
    features["corr_sector_60d"] = ret.rolling(corr_window, min_periods=corr_minp).corr(sector_ret)
    features["corr_change_20_60"] = features["corr_spy_20d"] - features["corr_spy_60d"]
    features["rolling_cov_spy"] = ret.rolling(corr_window, min_periods=corr_minp).cov(market_ret)
    corr_frame = features[["corr_spy_60d", "corr_qqq_60d", "corr_sector_60d"]]
    features["average_market_correlation"] = corr_frame.mean(axis=1)
    features["correlation_break_zscore"] = _zscore(features["corr_spy_60d"], 120)
    realised_vol = ret.rolling(20, min_periods=_min_periods(20, cfg)).std(ddof=0)
    features["systemic_correlation_score"] = pd.concat(
        [
            _zscore(features["average_market_correlation"], 120),
            _zscore(realised_vol, 120),
        ],
        axis=1,
    ).mean(axis=1)
    features["rolling_beta_instability"] = (
        features["market_model_beta"].rolling(beta_window, min_periods=beta_minp).std(ddof=0)
    )
    features["rolling_corr_instability"] = (
        features["average_market_correlation"]
        .rolling(corr_window, min_periods=corr_minp)
        .std(ddof=0)
    )


def _add_factor_ranks(features: pd.DataFrame, df: pd.DataFrame) -> None:
    close_columns = _available_close_columns(df)
    if len(close_columns) < 2:
        rank = pd.Series(0.5, index=df.index)
    else:
        momentum = pd.DataFrame(
            {column: _safe_log_return(df, column, 30) for column in close_columns},
            index=df.index,
        )
        rank = momentum.rank(axis=1, pct=True).get("Close", pd.Series(0.5, index=df.index))
    features["momentum_30d_rank"] = rank
    features["cross_sectional_rank_momentum"] = rank
    features["cross_sectional_rank_value"] = 0.5
    features["cross_sectional_rank_quality"] = 0.5
    features["rank_change_20d"] = rank - rank.shift(20)
    features["rank_percentile"] = rank.fillna(0.5)


def _add_pca_features(
    features: pd.DataFrame,
    df: pd.DataFrame,
    cfg: PipelineConfig,
    warning_rows: list[dict[str, object]],
) -> None:
    columns = _available_close_columns(df)
    pca_columns = [
        "pc1_market_mode_return",
        "pc1_explained_variance",
        "pc2_sector_rotation_score",
        "asset_loading_pc1",
        "asset_loading_pc2",
        "idiosyncratic_residual_from_pca",
        "pca_reconstruction_error",
        "pca_market_mode_flag",
    ]
    if len(columns) < 3 or importlib.util.find_spec("sklearn") is None:
        for column in pca_columns:
            features[column] = np.nan
        reason = "insufficient_series" if len(columns) < 3 else "missing_optional_dependency"
        warning_rows.append(
            _warning(reason, "PCA needs at least three valid return series.", "pca")
        )
        return

    from sklearn.decomposition import PCA

    returns = pd.DataFrame({column: _safe_log_return(df, column, 1) for column in columns})
    rows: dict[str, list[float]] = {column: [] for column in pca_columns}
    window = int(cfg.pca_window)
    minp = _min_periods(window, cfg)
    for pos in range(len(returns)):
        window_frame = returns.iloc[max(0, pos - window + 1) : pos + 1].dropna()
        if len(window_frame) < minp or window_frame.shape[1] < 3:
            for column in pca_columns:
                rows[column].append(np.nan)
            continue
        centered = window_frame - window_frame.mean()
        std = centered.std(ddof=0).replace(0.0, np.nan)
        standardized = (centered / std).dropna(axis=1)
        if standardized.shape[1] < 3:
            for column in pca_columns:
                rows[column].append(np.nan)
            continue
        pca = PCA(n_components=min(2, standardized.shape[1]))
        transformed = pca.fit_transform(standardized.values)
        reconstructed = pca.inverse_transform(transformed)
        asset_col = list(standardized.columns).index("Close") if "Close" in standardized else 0
        latest_return = float(window_frame.iloc[-1].get("Close", np.nan))
        latest_recon = reconstructed[-1, asset_col] * float(std.iloc[asset_col]) + float(
            window_frame.mean().iloc[asset_col]
        )
        rows["pc1_market_mode_return"].append(float(transformed[-1, 0]))
        rows["pc1_explained_variance"].append(float(pca.explained_variance_ratio_[0]))
        rows["pc2_sector_rotation_score"].append(
            float(transformed[-1, 1]) if transformed.shape[1] > 1 else np.nan
        )
        rows["asset_loading_pc1"].append(float(pca.components_[0, asset_col]))
        rows["asset_loading_pc2"].append(
            float(pca.components_[1, asset_col]) if pca.components_.shape[0] > 1 else np.nan
        )
        rows["idiosyncratic_residual_from_pca"].append(latest_return - latest_recon)
        rows["pca_reconstruction_error"].append(abs(latest_return - latest_recon))
        rows["pca_market_mode_flag"].append(float(pca.explained_variance_ratio_[0] > 0.55))

    for column, values in rows.items():
        features[column] = values


def _add_cointegration_features(
    features: pd.DataFrame,
    df: pd.DataFrame,
    cfg: PipelineConfig,
    warning_rows: list[dict[str, object]],
) -> None:
    peer_col = (
        "semiconductors_Close"
        if "semiconductors_Close" in df
        else "tech_Close" if "tech_Close" in df else "Benchmark_Close"
    )
    window = int(cfg.cointegration_window)
    minp = _min_periods(window, cfg)
    asset_log = np.log(df["Close"].astype(float).ffill())
    peer_log = np.log(df[peer_col].astype(float).ffill())
    beta = _rolling_beta(asset_log.diff(), peer_log.diff(), window, minp)
    spread = asset_log - beta * peer_log
    spread_mean = spread.rolling(window, min_periods=minp).mean()
    spread_std = spread.rolling(window, min_periods=minp).std(ddof=0)
    spread_z = (spread - spread_mean) / spread_std.replace(0.0, np.nan)
    features["sector_spread_zscore"] = spread_z
    features["peer_spread_zscore"] = spread_z
    features["spread_half_life"] = _rolling_half_life(spread, window, minp)
    features["spread_volatility"] = spread.rolling(window, min_periods=minp).std(ddof=0)
    features["spread_break_flag"] = (spread_z.abs() > 2.5).astype(float)
    if importlib.util.find_spec("statsmodels") is None:
        features["cointegration_pvalue_vs_sector"] = np.nan
        features["cointegration_pvalue_vs_peer"] = np.nan
        warning_rows.append(_warning("missing_optional_dependency", "statsmodels", "cointegration"))
        return
    from statsmodels.tsa.stattools import coint

    def coint_pvalue(values: np.ndarray) -> float:
        n = len(values) // 2
        asset = values[:n]
        peer = values[n:]
        valid = np.isfinite(asset) & np.isfinite(peer)
        if valid.sum() < minp:
            return np.nan
        try:
            return float(coint(asset[valid], peer[valid])[1])
        except Exception:
            return np.nan

    joined = pd.concat([asset_log, peer_log], axis=1)
    packed = pd.Series(index=df.index, dtype=object)
    pvalues = []
    for pos in range(len(joined)):
        frame = joined.iloc[max(0, pos - window + 1) : pos + 1]
        if len(frame.dropna()) < minp:
            pvalues.append(np.nan)
            continue
        values = np.r_[frame.iloc[:, 0].values, frame.iloc[:, 1].values]
        packed.iloc[pos] = values
        pvalues.append(coint_pvalue(values))
    features["cointegration_pvalue_vs_sector"] = pvalues
    features["cointegration_pvalue_vs_peer"] = pvalues


def _add_parameter_instability(features: pd.DataFrame) -> None:
    pieces = [
        _zscore(features.get("rolling_mean_instability", pd.Series(index=features.index)).abs()),
        _zscore(features.get("rolling_vol_instability", pd.Series(index=features.index)).abs()),
        _zscore(features.get("rolling_beta_instability", pd.Series(index=features.index)).abs()),
        _zscore(features.get("rolling_corr_instability", pd.Series(index=features.index)).abs()),
        _zscore(features.get("rolling_ar1_instability", pd.Series(index=features.index)).abs()),
    ]
    features["parameter_instability_z"] = pd.concat(pieces, axis=1).mean(axis=1)


def _enabled(feature_set: Iterable[str], name: str) -> bool:
    return name in set(feature_set)


def build_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
    warnings: list[StructuredWarning] | None = None,
) -> pd.DataFrame:
    """Build trailing-only statistical features inspired by research_public lectures."""
    result = build_research_public_features(base, cfg, provider)
    if warnings is not None and not result.warnings.empty:
        for row in result.warnings.to_dict("records"):
            warnings.append(
                StructuredWarning(
                    code=str(row.get("reason", "research_public_warning")),
                    severity="warning",
                    module="feature_blocks.research_public",
                    message=str(row.get("detail", "")),
                    details=row,
                )
            )
    return result.features


def build_research_public_features(
    base: pd.DataFrame,
    cfg: PipelineConfig,
    provider: BaseDataProvider | None = None,
) -> FeatureBlockResult:
    feature_set = set(cfg.research_public_feature_set)
    features = pd.DataFrame(index=base.index)
    warning_rows: list[dict[str, object]] = []
    ret = base["ret_1"] if "ret_1" in base else _safe_log_return(base, "Close", 1)
    log_price = np.log(base["Close"].astype(float).ffill())

    if _enabled(feature_set, "stationarity"):
        _add_stationarity(features, log_price, ret, cfg, warning_rows)
    if _enabled(feature_set, "autoregressive"):
        _add_autoregressive(features, ret, cfg)
    if _enabled(feature_set, "volatility_clustering"):
        _add_volatility_clustering(features, ret, cfg)
    if (
        _enabled(feature_set, "market_residual")
        or _enabled(feature_set, "regression_diagnostics")
        or _enabled(feature_set, "correlation_regime")
    ):
        if (
            _enabled(feature_set, "regression_diagnostics")
            and importlib.util.find_spec("scipy") is None
        ):
            warning_rows.append(
                _warning("missing_optional_dependency", "scipy", "jarque_bera_pvalue")
            )
        _add_market_residual_and_correlation(features, base, ret, cfg)
    if _enabled(feature_set, "factor_ranks"):
        _add_factor_ranks(features, base)
    if _enabled(feature_set, "pca_market_mode"):
        _add_pca_features(features, base, cfg, warning_rows)
    if _enabled(feature_set, "cointegration_spread"):
        _add_cointegration_features(features, base, cfg, warning_rows)
    if _enabled(feature_set, "instability"):
        _add_parameter_instability(features)

    features = features.replace([np.inf, -np.inf], np.nan)
    warnings_frame = pd.DataFrame(warning_rows)
    return FeatureBlockResult(
        name=RESEARCH_PUBLIC_GROUP,
        features=features,
        metadata=feature_metadata(RESEARCH_PUBLIC_GROUP, list(features.columns)),
        warnings=warnings_frame,
        produced_columns=list(features.columns),
    )
