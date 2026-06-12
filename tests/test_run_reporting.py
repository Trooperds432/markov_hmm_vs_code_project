from __future__ import annotations

import json

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.run_reporting import write_run_audit_outputs


def test_run_summary_and_output_status_mark_disabled_files(tmp_path):
    (tmp_path / "nn_predictions.csv").write_text("x\n1\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "feature": "z_ret_1",
                "used_in_hmm": True,
                "used_in_hmm_requested": True,
                "drop_reason": "",
            }
        ]
    ).to_csv(tmp_path / "feature_manifest.csv", index=False)
    cfg = PipelineConfig(enable_nn_confirmation=False, enable_experiment_tracking=False)

    write_run_audit_outputs(cfg, tmp_path, mode="research", raw_rows=10, feature_rows=8)

    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    status = pd.read_csv(tmp_path / "output_status.csv")
    modules = pd.read_csv(tmp_path / "active_modules.csv")

    assert summary["feature_counts"]["hmm_features_used"] == 1
    assert summary["active_modules"]["nn_confirmation"] is False
    nn_status = status.set_index("file").loc["nn_predictions.csv"]
    assert nn_status["status"] == "skipped_disabled"
    assert not bool(modules.set_index("module").loc["nn_confirmation", "active"])
