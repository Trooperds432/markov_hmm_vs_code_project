from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.hmm_model import convergence_history, model_information_criteria
from src.markov_hmm_trader.model_selection import (
    hmm_num_parameters,
    information_criteria,
    select_hmm_model,
    validation_split,
)


def _selection_frame(rows: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    ret = np.r_[rng.normal(0.01, 0.002, rows // 2), rng.normal(-0.01, 0.002, rows - rows // 2)]
    return pd.DataFrame(
        {
            "ret_1": ret,
            "alpha_score_long": (ret > 0).astype(int) * 5,
            "feature_a": ret,
            "feature_b": np.roll(ret, 1),
        },
        index=dates,
    )


def test_validation_split_keeps_last_rows_for_validation():
    frame = _selection_frame(100)

    inner, validation = validation_split(frame, 0.2)

    assert len(inner) == 80
    assert len(validation) == 20
    assert inner.index.max() < validation.index.min()


def test_information_criteria_and_parameter_count():
    params = hmm_num_parameters(3, 2, "diag")
    aic, bic = information_criteria(-100.0, params, 50)

    assert params > 0
    assert aic > 0
    assert bic > aic


def test_native_information_criteria_preferred_when_available():
    model = SimpleNamespace(
        aic=lambda X: 12.0,
        bic=lambda X: 15.0,
        score=lambda X: -100.0,
    )

    aic, bic, source = model_information_criteria(
        model,
        np.ones((10, 2)),
        n_parameters=3,
        use_native=True,
    )

    assert aic == 12.0
    assert bic == 15.0
    assert source == "hmmlearn_native"


def test_information_criteria_falls_back_when_native_unavailable():
    model = SimpleNamespace(score=lambda X: -100.0)

    aic, bic, source = model_information_criteria(
        model,
        np.ones((10, 2)),
        n_parameters=3,
        use_native=True,
    )

    assert aic == 206.0
    assert bic > aic
    assert source == "local_fallback"


def test_convergence_history_reads_monitor_history():
    model = SimpleNamespace(monitor_=SimpleNamespace(history=[-3.0, -2.0, -1.5]))

    assert convergence_history(model) == (-3.0, -2.0, -1.5)


def test_select_hmm_model_marks_one_candidate_selected():
    frame = _selection_frame()
    cfg = PipelineConfig(
        n_states=2,
        random_seeds=(1,),
        n_iter=5,
        covariance_type="diag",
        enable_hmm_model_selection=True,
        hmm_state_candidates=(2,),
        hmm_covariance_candidates=("diag",),
    )

    fit = select_hmm_model(frame, ["feature_a", "feature_b"], cfg)

    assert fit.selected_n_states == 2
    assert len(fit.model_selection_rows) == 1
    assert fit.model_selection_rows[0]["selected"] is True
