from __future__ import annotations

from datetime import datetime

from src.markov_hmm_trader.config import PipelineConfig, config_hash
from src.markov_hmm_trader.experiment import create_run_folder


def test_run_folder_creation_and_config_hash(tmp_path):
    cfg = PipelineConfig(experiment_name="unit_test")

    run_dir = create_run_folder(cfg, tmp_path, timestamp=datetime(2024, 1, 2, 3, 4, 5))

    assert run_dir.exists()
    assert run_dir.name == "20240102_030405_unit_test"
    assert len(config_hash(cfg)) == 16
