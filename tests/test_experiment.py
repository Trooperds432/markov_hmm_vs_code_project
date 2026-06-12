from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig, config_hash
from src.markov_hmm_trader.experiment import create_run_folder, finalize_experiment_run


def test_run_folder_creation_and_config_hash(tmp_path):
    cfg = PipelineConfig(experiment_name="unit_test")

    run_dir = create_run_folder(cfg, tmp_path, timestamp=datetime(2024, 1, 2, 3, 4, 5))

    assert run_dir.exists()
    assert run_dir.name == "20240102_030405_unit_test"
    assert len(config_hash(cfg)) == 16


def test_finalize_experiment_writes_stage2e_artifacts(tmp_path):
    cfg = PipelineConfig(
        experiment_name="unit_test",
        model_registry_path=str(tmp_path / "registry" / "model_registry.csv"),
    )
    run_dir = create_run_folder(cfg, tmp_path / "runs", timestamp=datetime(2024, 1, 2, 3, 4, 5))
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    pd.DataFrame([{"feature": "z_ret_1", "used_in_hmm": True}]).to_csv(
        output_dir / "feature_manifest.csv",
        index=False,
    )
    pd.DataFrame([{"code": "test_warning", "severity": "warning"}]).to_csv(
        output_dir / "warnings.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "window_id": 1,
                "candidate_id": 1,
                "selected": True,
                "validation_log_likelihood": 1.0,
            }
        ]
    ).to_csv(output_dir / "hmm_model_selection.csv", index=False)
    stats = pd.Series({"strategy_sharpe": 1.0, "strategy_cagr": 0.1})

    finalize_experiment_run(cfg, run_dir, output_dir, stats, {"model_type": "GaussianHMM"})

    for name in [
        "metrics.json",
        "artifacts_index.json",
        "dependency_report.json",
        "warnings.json",
        "feature_manifest.csv",
        "model_selection_candidates.csv",
        "experiment_manifest.json",
    ]:
        assert (run_dir / name).exists()
    assert (output_dir / "artifacts_index.json").exists()
    assert (output_dir / "experiment_metadata.json").exists()
