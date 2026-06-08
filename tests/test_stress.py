from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.stress import (
    block_bootstrap_returns,
    make_cost_slippage_stress,
)


def test_block_bootstrap_returns_preserves_length():
    returns = pd.Series([0.01, -0.02, 0.03, 0.0])

    paths = block_bootstrap_returns(returns, paths=3, block_size=2, random_seed=1)

    assert len(paths) == 3
    assert all(len(path) == len(returns) for path in paths)


def test_cost_slippage_grid_shape():
    df = pd.DataFrame(
        {
            "ret_1": [0.0, 0.01, -0.01],
            "position": [0.0, 1.0, 1.0],
        }
    )
    cfg = PipelineConfig(
        stress_commission_bps_values=(1.0, 2.0),
        stress_slippage_bps_values=(3.0, 4.0),
    )

    result = make_cost_slippage_stress(df, cfg)

    assert len(result) == 4
    assert {"commission_bps", "slippage_bps"}.issubset(result.columns)
