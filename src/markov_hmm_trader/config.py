from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple
import json


@dataclass
class PipelineConfig:
    # Market data
    ticker: str = "NVDA"
    benchmark: str = "QQQ"
    start: str = "2012-01-01"
    end: Optional[str] = None
    interval: str = "1d"

    # HMM
    n_states: int = 3
    train_window: int = 756
    retrain_frequency: int = 21
    n_iter: int = 500
    tolerance: float = 1e-4
    covariance_type: str = "full"
    random_seeds: Tuple[int, ...] = (7, 13, 21, 42, 99, 123, 777)

    # Signal thresholds
    long_only: bool = True
    min_bull_prob: float = 0.42
    max_bear_prob_for_long: float = 0.45
    min_signal: float = 0.05
    min_alpha_score: int = 4

    # Exposure sizing
    base_exposure: float = 0.35
    core_exposure: float = 0.55
    max_exposure: float = 0.95
    bear_regime_exposure_mult: float = 0.20
    side_regime_exposure_mult: float = 0.60
    bull_regime_exposure_mult: float = 1.00

    # Volatility targeting
    use_vol_targeting: bool = True
    target_ann_vol: float = 0.28
    vol_lookback: int = 30
    min_vol_mult: float = 0.35
    max_vol_mult: float = 1.75

    # Drawdown throttle
    use_drawdown_throttle: bool = True
    dd_level_1: float = 0.10
    dd_level_2: float = 0.22
    dd_mult_1: float = 0.70
    dd_mult_2: float = 0.40

    # Costs
    commission_bps: float = 7.5
    slippage_bps: float = 3.0

    # Trend and relative strength
    fast_ema: int = 50
    slow_ema: int = 200
    rs_fast: int = 20
    rs_slow: int = 60

    # Reporting
    bars_per_year: int = 252

    # Diagnostics and stress testing
    enable_diagnostics: bool = True
    enable_stress_tests: bool = True
    enable_benchmark: bool = False
    stress_cost_multipliers: Tuple[float, ...] = (2.0, 3.0)
    stress_execution_lag_bars: Tuple[int, ...] = (1, 2)
    stress_bootstrap_paths: int = 200
    stress_bootstrap_block_size: int = 21
    stress_random_seed: int = 1234
    benchmark_maxiter: int = 200


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    valid_names = {field.name for field in fields(PipelineConfig)}
    clean = {k: v for k, v in raw.items() if k in valid_names}

    tuple_fields = {
        "random_seeds",
        "stress_cost_multipliers",
        "stress_execution_lag_bars",
    }
    for name in tuple_fields:
        if name in clean and isinstance(clean[name], list):
            clean[name] = tuple(clean[name])

    return PipelineConfig(**clean)
