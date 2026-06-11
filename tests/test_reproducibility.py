from __future__ import annotations

import json
import random

import numpy as np

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.reproducibility import (
    make_config_hash,
    set_global_seed,
    write_experiment_metadata,
)


def test_global_seed_reproducibility():
    set_global_seed(123)
    first = (random.random(), np.random.random())
    set_global_seed(123)
    second = (random.random(), np.random.random())

    assert first == second


def test_config_hash_is_stable():
    cfg = PipelineConfig(random_seed=42)

    assert make_config_hash(cfg) == make_config_hash(cfg)


def test_experiment_metadata_json_contract(tmp_path):
    path = tmp_path / "experiment_metadata.json"

    metadata = write_experiment_metadata(path, PipelineConfig(random_seed=7))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["random_seed"] == 7
    assert payload["config_hash"] == metadata["config_hash"]
    assert "git_commit" in payload
