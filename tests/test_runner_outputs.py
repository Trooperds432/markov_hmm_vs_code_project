from __future__ import annotations

import numpy as np
import pandas as pd

from src.markov_hmm_trader.backtest import BacktestResult
from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.runner import run_single
from src.markov_hmm_trader.states import STATE_BEAR, STATE_BULL, STATE_SIDE


def test_run_single_writes_stage2a_additive_outputs(monkeypatch, tmp_path):
    dates = pd.bdate_range("2020-01-01", periods=40)
    returns = pd.Series(np.full(len(dates), 0.001), index=dates)
    backtest = pd.DataFrame(
        {
            "Close": 100.0 * np.exp(returns.cumsum()),
            "ret_1": returns,
            STATE_BULL: 0.7,
            STATE_BEAR: 0.1,
            STATE_SIDE: 0.2,
            "hidden_state": STATE_BULL,
            "position": 0.5,
            "hmm_signal": 0.6,
            "strategy_ret": returns * 0.5,
            "strategy_equity": (1.0 + returns * 0.5).cumprod(),
            "buy_hold_equity": (1.0 + returns).cumprod(),
        },
        index=dates,
    )
    result = BacktestResult(
        backtest=backtest,
        window_diagnostics=pd.DataFrame([{"window_id": 1, "test_log_likelihood": 8.0}]),
        seed_diagnostics=pd.DataFrame(),
        hmm_model_selection=pd.DataFrame(
            [
                {
                    "window_id": 1,
                    "candidate_id": 1,
                    "train_log_likelihood": 10.0,
                    "validation_log_likelihood": 9.0,
                }
            ]
        ),
        hmm_emissions=pd.DataFrame(),
        hmm_transition_matrices=pd.DataFrame(),
        hmm_state_stability=pd.DataFrame(),
        hmm_stationary_distribution=pd.DataFrame(),
        hmm_decode_comparison=pd.DataFrame(),
        hmm_component_diagnostics=pd.DataFrame(),
        hmm_synthetic_sample_stress=pd.DataFrame(),
        hmm_synthetic_sample_stress_summary=pd.DataFrame(),
    )

    monkeypatch.setattr("src.markov_hmm_trader.runner.load_market_data", lambda cfg: backtest)
    monkeypatch.setattr("src.markov_hmm_trader.runner.make_features", lambda raw, cfg: backtest)
    monkeypatch.setattr(
        "src.markov_hmm_trader.runner.walk_forward_backtest", lambda df, cfg: result
    )
    monkeypatch.setattr(
        "src.markov_hmm_trader.runner.add_backtest_metric_columns", lambda df, cfg: df
    )
    monkeypatch.setattr(
        "src.markov_hmm_trader.runner.make_report",
        lambda df, cfg: pd.Series({"ticker": cfg.ticker, "strategy_sharpe": 1.0}),
    )
    monkeypatch.setattr(
        "src.markov_hmm_trader.runner.start_experiment_run", lambda cfg, output_dir: output_dir
    )
    monkeypatch.setattr(
        "src.markov_hmm_trader.runner.finalize_experiment_run",
        lambda cfg, experiment_dir, output_dir, stats, selected_model: None,
    )
    cfg = PipelineConfig(
        enable_observable_markov=True,
        enable_volatility_forecasts=True,
        enable_experiment_tracking=False,
        enable_stress_tests=False,
        enable_overfit_report=True,
        vol_model_candidates=("realised_vol", "ewma_vol"),
    )

    run_single(cfg, tmp_path, make_plots=False)

    assert (tmp_path / "volatility_forecasts.csv").exists()
    assert (tmp_path / "observable_markov_baseline.csv").exists()
    assert (tmp_path / "observable_markov_stationary_distribution.csv").exists()
    assert (tmp_path / "model_overfit_report.csv").exists()
    assert (tmp_path / "warnings.csv").exists()
    assert (tmp_path / "feature_manifest.csv").exists()
    assert (tmp_path / "data_provider_metadata.csv").exists()
    assert (tmp_path / "data_provider_warnings.csv").exists()
    assert (tmp_path / "feature_group_metadata.csv").exists()
    assert (tmp_path / "feature_group_warnings.csv").exists()
