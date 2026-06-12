from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .alt_data.overlays import adjusted_portfolio_score
from .config import PipelineConfig
from .data import load_market_data
from .factor_validation import factor_rank_diagnostics, long_short_spread_diagnostics
from .features import make_features
from .metrics import cagr, max_drawdown, sharpe
from .runner import run_single
from .states import STATE_BEAR, STATE_BULL


def allocate_weights(
    scores: pd.Series,
    volatility: pd.Series,
    cfg: PipelineConfig,
) -> pd.Series:
    mode = str(cfg.portfolio_mode).lower()
    max_positions = 1 if mode == "single_asset" else int(cfg.portfolio_max_positions)
    selected = scores.sort_values(ascending=False).head(max_positions)
    if selected.empty:
        return pd.Series(dtype=float)

    method = "risk_budgeted" if mode == "risk_budgeted" else cfg.portfolio_weighting_method
    if method == "equal":
        raw = pd.Series(1.0, index=selected.index)
    elif method in {"signal_weighted", "score"}:
        raw = selected.clip(lower=0.0)
    elif method in {"risk_budgeted", "risk_parity_simple", "vol_inverse", "inverse_vol"}:
        raw = 1.0 / volatility.reindex(selected.index).replace(0.0, np.nan)
    elif method == "score_vol_inverse":
        vol_inverse = 1.0 / volatility.reindex(selected.index).replace(0.0, np.nan)
        raw = selected.clip(lower=0.0) * vol_inverse
    else:
        raw = 1.0 / volatility.reindex(selected.index).replace(0.0, np.nan)

    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if raw.sum() <= 0.0:
        raw = pd.Series(1.0, index=selected.index)

    weights = raw / raw.sum()
    if method == "risk_budgeted":
        annual_vol = volatility.reindex(selected.index).replace([np.inf, -np.inf], np.nan)
        annual_vol = annual_vol.fillna(annual_vol.median()).fillna(0.0) * np.sqrt(cfg.bars_per_year)
        portfolio_vol = float(np.sqrt(np.square(weights * annual_vol).sum()))
        if portfolio_vol > 0.0:
            vol_scale = min(1.0, float(cfg.portfolio_target_vol) / portfolio_vol)
            weights = weights * vol_scale

    weights = weights.clip(upper=float(cfg.portfolio_max_weight_per_asset))
    max_invested = max(0.0, 1.0 - float(cfg.portfolio_cash_buffer))
    if weights.sum() > max_invested:
        weights = weights / weights.sum() * max_invested
    total = weights.sum()
    if total > 1.0:
        weights = weights / total
    return weights


def _rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    rule = "W-FRI" if frequency == "weekly" else "ME"
    return pd.Series(index, index=index).resample(rule).last().dropna().values


def _load_sector_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    sector_path = Path(path)
    if not sector_path.exists():
        return {}
    frame = pd.read_csv(sector_path)
    if not {"symbol", "sector"}.issubset(frame.columns):
        return {}
    return dict(zip(frame["symbol"].astype(str), frame["sector"].astype(str), strict=False))


def _portfolio_corr_rows(date: pd.Timestamp, corr: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for left in corr.columns:
        for right in corr.columns:
            if str(left) >= str(right):
                continue
            rows.append(
                {
                    "date": date.date().isoformat(),
                    "symbol_a": left,
                    "symbol_b": right,
                    "correlation": float(corr.loc[left, right]),
                }
            )
    return rows


def _apply_portfolio_constraints(
    date: pd.Timestamp,
    raw_weights: pd.Series,
    previous: pd.Series,
    raw_scores: pd.Series,
    returns: pd.DataFrame,
    cfg: PipelineConfig,
    sector_map: dict[str, str],
) -> tuple[
    pd.Series,
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, object]],
]:
    symbols = list(previous.index)
    target = raw_weights.reindex(symbols).fillna(0.0).clip(lower=0.0)
    old = previous.reindex(symbols).fillna(0.0)
    scores = raw_scores.reindex(symbols).fillna(0.0)
    constraint_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []
    adjustments: dict[str, list[str]] = {symbol: [] for symbol in symbols}

    for symbol in symbols:
        if (
            old[symbol] <= 0.0
            and target[symbol] > 0.0
            and scores[symbol] < cfg.portfolio_allow_new_position_threshold
        ):
            target[symbol] = 0.0
            adjustments[symbol].append("new_position_score_below_threshold")
        if old[symbol] > 0.0 and scores[symbol] < cfg.portfolio_exit_score_threshold:
            target[symbol] = 0.0
            adjustments[symbol].append("exit_score_threshold")

    corr = returns.loc[:date].tail(cfg.portfolio_corr_window).corr().fillna(0.0)
    if cfg.portfolio_use_correlation_penalty and not corr.empty:
        for left in symbols:
            for right in symbols:
                if left >= right or left not in corr or right not in corr:
                    continue
                value = float(corr.loc[left, right])
                if (
                    value > cfg.portfolio_max_pairwise_corr
                    and target[left] > 0.0
                    and target[right] > 0.0
                ):
                    loser = left if scores[left] <= scores[right] else right
                    target[loser] *= 0.5
                    adjustments[loser].append("correlation_penalty")
                    constraint_rows.append(
                        {
                            "date": date.date().isoformat(),
                            "constraint": "max_pairwise_corr",
                            "triggered": True,
                            "symbol": loser,
                            "detail": f"{left}/{right} corr={value:.3f}",
                        }
                    )
        corr_rows = _portfolio_corr_rows(date, corr)

    sector_breaches = 0
    if sector_map:
        sectors = pd.Series({symbol: sector_map.get(symbol, "unknown") for symbol in symbols})
        for sector, members in sectors.groupby(sectors).groups.items():
            member_list = list(members)
            total = float(target.reindex(member_list).sum())
            if total > cfg.portfolio_sector_cap:
                scale = cfg.portfolio_sector_cap / total
                target.loc[member_list] *= scale
                sector_breaches += 1
                for symbol in member_list:
                    adjustments[symbol].append("sector_cap_scaled")
                constraint_rows.append(
                    {
                        "date": date.date().isoformat(),
                        "constraint": "sector_cap",
                        "triggered": True,
                        "symbol": ",".join(member_list),
                        "detail": f"{sector} scaled to cap {cfg.portfolio_sector_cap}",
                    }
                )
    elif cfg.portfolio_sector_map_path:
        constraint_rows.append(
            {
                "date": date.date().isoformat(),
                "constraint": "sector_map",
                "triggered": False,
                "symbol": "",
                "detail": "sector_map_missing_or_invalid",
            }
        )

    target = target.clip(upper=float(cfg.portfolio_max_weight_per_asset))
    max_invested = max(0.0, 1.0 - float(cfg.portfolio_cash_buffer))
    if target.sum() > max_invested:
        target = target / target.sum() * max_invested
        for symbol in symbols:
            adjustments[symbol].append("cash_buffer_scaled")

    changes = target - old
    small = changes.abs() < cfg.portfolio_min_rebalance_change
    target.loc[small] = old.loc[small]
    for symbol in list(pd.Index(symbols)[small.to_numpy()]):
        adjustments[symbol].append("min_rebalance_change_hold")

    turnover = float((target - old).abs().sum())
    if turnover > cfg.portfolio_max_turnover_per_rebalance > 0.0:
        scale = cfg.portfolio_max_turnover_per_rebalance / turnover
        target = old + (target - old) * scale
        turnover = float((target - old).abs().sum())
        for symbol in symbols:
            adjustments[symbol].append("turnover_cap_scaled")
        constraint_rows.append(
            {
                "date": date.date().isoformat(),
                "constraint": "max_turnover_per_rebalance",
                "triggered": True,
                "symbol": "",
                "detail": f"scaled turnover to {cfg.portfolio_max_turnover_per_rebalance}",
            }
        )

    risk_window = returns.loc[:date].tail(cfg.portfolio_corr_window)
    port_ret = (risk_window.fillna(0.0) * target.reindex(risk_window.columns).fillna(0.0)).sum(
        axis=1
    )
    corr_values = (
        corr.where(~np.eye(len(corr), dtype=bool)).stack()
        if not corr.empty
        else pd.Series(dtype=float)
    )
    risk_row = {
        "date": date.date().isoformat(),
        "gross_exposure": float(target.abs().sum()),
        "net_exposure": float(target.sum()),
        "cash_weight": float(max(0.0, 1.0 - target.sum())),
        "number_of_holdings": int((target > 0.0).sum()),
        "concentration_hhi": float(target.pow(2).sum()),
        "max_weight": float(target.max()) if len(target) else 0.0,
        "turnover": turnover,
        "estimated_portfolio_vol": float(port_ret.std(ddof=0) * np.sqrt(cfg.bars_per_year)),
        "average_pairwise_corr": float(corr_values.mean()) if not corr_values.empty else 0.0,
        "max_pairwise_corr": float(corr_values.max()) if not corr_values.empty else 0.0,
        "sector_cap_breaches": int(sector_breaches),
        "constraints_triggered": ",".join(
            sorted({item for values in adjustments.values() for item in values if item})
        ),
    }
    decisions = []
    for symbol in symbols:
        final = float(target[symbol])
        reason = "hold"
        if final > old[symbol]:
            reason = "buy"
        elif final < old[symbol]:
            reason = "sell"
        decisions.append(
            {
                "date": date.date().isoformat(),
                "symbol": symbol,
                "old_weight": float(old[symbol]),
                "target_weight": float(raw_weights.reindex(symbols).fillna(0.0)[symbol]),
                "final_weight": final,
                "raw_score": float(raw_scores.get(symbol, np.nan)),
                "adjusted_score": float(scores[symbol]),
                "reason": reason,
                "constraint_adjustment": ",".join(dict.fromkeys(adjustments[symbol])),
            }
        )
    return target, decisions, constraint_rows, risk_row, corr_rows


def run_portfolio(cfg: PipelineConfig, output_dir) -> dict[str, pd.DataFrame | pd.Series]:
    output_dir.mkdir(parents=True, exist_ok=True)
    asset_backtests = {}
    signal_rows = []

    for symbol in cfg.portfolio_symbols:
        local_cfg = replace(
            cfg,
            ticker=symbol,
            benchmark=cfg.portfolio_benchmark,
            enable_diagnostics=False,
            enable_stress_tests=False,
            enable_experiment_tracking=False,
        )
        try:
            backtest, _ = run_single(local_cfg, output_dir / symbol, make_plots=False)
            asset_backtests[symbol] = backtest
            latest = backtest.iloc[-1]
            signal_rows.append(
                {
                    "symbol": symbol,
                    "bull_probability": latest.get(STATE_BULL, np.nan),
                    "bear_probability": latest.get(STATE_BEAR, np.nan),
                    "trend_score": latest.get("alpha_score_long", np.nan),
                    "relative_strength": latest.get("rel_ret_20", np.nan),
                    "glassdoor_stress_z": latest.get("glassdoor_stress_z", np.nan),
                    "wsb_mentions_z_20d": latest.get("wsb_mentions_z_20d", np.nan),
                    "wsb_meme_risk_score": latest.get("wsb_meme_risk_score", np.nan),
                    "margin_of_safety": latest.get("margin_of_safety", np.nan),
                    "valuation_percentile": latest.get("valuation_percentile", np.nan),
                    "alt_data_risk_score": latest.get("alt_data_risk_score", np.nan),
                    "alt_data_confidence_score": latest.get("alt_data_confidence_score", np.nan),
                    "adjusted_portfolio_score": adjusted_portfolio_score(latest, cfg),
                }
            )
        except Exception as err:
            signal_rows.append({"symbol": symbol, "error": type(err).__name__})

    if not asset_backtests:
        empty = pd.DataFrame()
        return {
            "portfolio_equity_curve": empty,
            "portfolio_weights": empty,
            "portfolio_trades": empty,
            "portfolio_stats": pd.Series(dtype=float),
            "portfolio_asset_signals": pd.DataFrame(signal_rows),
            "portfolio_risk_report": empty,
            "portfolio_rebalance_decisions": empty,
            "portfolio_correlation_matrix": empty,
            "portfolio_constraint_report": empty,
        }

    returns = pd.DataFrame({symbol: bt["ret_1"] for symbol, bt in asset_backtests.items()}).dropna(
        how="all"
    )
    scores = pd.DataFrame(
        {
            symbol: bt[STATE_BULL].fillna(0.0) - bt[STATE_BEAR].fillna(0.0)
            for symbol, bt in asset_backtests.items()
        }
    ).reindex(returns.index)
    if cfg.use_alt_data_portfolio_ranking:
        scores = pd.DataFrame(
            {
                symbol: bt.apply(lambda row: adjusted_portfolio_score(row, cfg), axis=1)
                for symbol, bt in asset_backtests.items()
            }
        ).reindex(returns.index)
    volatility = returns.rolling(63).std(ddof=0).fillna(returns.expanding().std(ddof=0))
    weights = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    previous_weights = pd.Series(0.0, index=returns.columns)
    sector_map = _load_sector_map(cfg.portfolio_sector_map_path)
    decision_rows: list[dict[str, object]] = []
    constraint_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    corr_rows: list[dict[str, object]] = []

    for date in _rebalance_dates(returns.index, cfg.portfolio_rebalance_frequency):
        date = pd.Timestamp(date)
        if date not in returns.index:
            continue
        historical_scores = scores.loc[:date].iloc[-1]
        historical_vol = volatility.loc[:date].iloc[-1]
        allocated = allocate_weights(historical_scores, historical_vol, cfg)
        allocated, decisions, constraints, risk_row, correlations = _apply_portfolio_constraints(
            date,
            allocated,
            previous_weights,
            historical_scores,
            returns,
            cfg,
            sector_map,
        )
        weights.loc[date] = 0.0
        weights.loc[date, allocated.index] = allocated
        previous_weights = allocated.reindex(weights.columns).fillna(0.0)
        decision_rows.extend(decisions)
        constraint_rows.extend(constraints)
        risk_rows.append(risk_row)
        corr_rows.extend(correlations)

    weights = weights.ffill().fillna(0.0)
    portfolio_returns = (weights.shift(1).fillna(0.0) * returns.fillna(0.0)).sum(axis=1)
    equity = (1.0 + portfolio_returns).cumprod()
    equity_curve = pd.DataFrame(
        {
            "portfolio_ret": portfolio_returns,
            "portfolio_equity": equity,
            "holdings": (weights > 0.0).sum(axis=1),
            "concentration": (weights.pow(2)).sum(axis=1),
        }
    )
    trades = weights.diff().abs().fillna(weights.abs())
    stats = pd.Series(
        {
            "portfolio_cagr": cagr(equity, cfg.bars_per_year),
            "portfolio_sharpe": sharpe(portfolio_returns, cfg.bars_per_year),
            "portfolio_max_drawdown": max_drawdown(equity),
            "turnover": float(trades.sum(axis=1).mean()),
            "average_number_of_holdings": float(equity_curve["holdings"].mean()),
            "concentration": float(equity_curve["concentration"].mean()),
            "weights_sum_max": float(weights.sum(axis=1).max()),
            "max_weight": float(weights.max(axis=1).max()),
        }
    )

    outputs = {
        "portfolio_equity_curve": equity_curve,
        "portfolio_weights": weights,
        "portfolio_trades": trades,
        "portfolio_stats": stats,
        "portfolio_asset_signals": pd.DataFrame(signal_rows),
        "portfolio_risk_report": pd.DataFrame(risk_rows),
        "portfolio_rebalance_decisions": pd.DataFrame(decision_rows),
        "portfolio_correlation_matrix": pd.DataFrame(corr_rows),
        "portfolio_constraint_report": pd.DataFrame(
            constraint_rows
            or [
                {
                    "date": "",
                    "constraint": "none",
                    "triggered": False,
                    "symbol": "",
                    "detail": "no_constraints_triggered",
                }
            ]
        ),
    }
    equity_curve.to_csv(output_dir / "portfolio_equity_curve.csv")
    weights.to_csv(output_dir / "portfolio_weights.csv")
    trades.to_csv(output_dir / "portfolio_trades.csv")
    stats.to_frame("value").to_csv(output_dir / "portfolio_stats.csv")
    outputs["portfolio_asset_signals"].to_csv(
        output_dir / "portfolio_asset_signals.csv", index=False
    )
    outputs["portfolio_risk_report"].to_csv(output_dir / "portfolio_risk_report.csv", index=False)
    outputs["portfolio_rebalance_decisions"].to_csv(
        output_dir / "portfolio_rebalance_decisions.csv",
        index=False,
    )
    outputs["portfolio_correlation_matrix"].to_csv(
        output_dir / "portfolio_correlation_matrix.csv",
        index=False,
    )
    outputs["portfolio_constraint_report"].to_csv(
        output_dir / "portfolio_constraint_report.csv",
        index=False,
    )
    if cfg.enable_research_public_features:
        rank_source = equity_curve.copy()
        rank_source["rank_percentile"] = scores.rank(axis=1, pct=True).mean(axis=1)
        rank_source["ret_1"] = portfolio_returns
        factor_rank_diagnostics(rank_source, cfg).to_csv(
            output_dir / "factor_rank_diagnostics.csv",
            index=False,
        )
        long_short_spread_diagnostics(rank_source, cfg).to_csv(
            output_dir / "long_short_spread_diagnostics.csv",
            index=False,
        )
    return outputs


def load_portfolio_market_data(cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    data = {}
    for symbol in cfg.portfolio_symbols:
        local_cfg = replace(cfg, ticker=symbol, benchmark=cfg.portfolio_benchmark)
        data[symbol] = make_features(load_market_data(local_cfg), local_cfg)
    return data
