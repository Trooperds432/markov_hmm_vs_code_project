from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.model_audit import (
    MODEL_SELECTION_AUDIT_COLUMNS,
    build_model_selection_candidates,
)


def test_model_selection_audit_contract_and_selected_candidate():
    candidates = pd.DataFrame(
        [
            {
                "window_id": 1,
                "candidate_id": 1,
                "model_type": "gaussian",
                "n_states": 3,
                "n_mix": 1,
                "covariance_type": "diag",
                "train_log_likelihood": 10.0,
                "validation_log_likelihood": 8.0,
                "validation_aic": 1.0,
                "validation_bic": 2.0,
                "selection_score": 8.0,
                "selected": True,
                "converged": True,
            },
            {
                "window_id": 1,
                "candidate_id": 2,
                "model_type": "gaussian",
                "n_states": 4,
                "n_mix": 1,
                "covariance_type": "diag",
                "selection_score": 7.0,
                "selected": False,
                "converged": False,
            },
        ]
    )

    audit = build_model_selection_candidates(candidates, "SYNTH", "2020-01-01T00:00:00Z")

    assert list(audit.columns) == MODEL_SELECTION_AUDIT_COLUMNS
    assert audit["selected"].sum() == 1
    assert audit.loc[1, "rejection_reason"] == "lower_selection_score"
    assert "non_converged" in audit.loc[1, "warning_flags"]
