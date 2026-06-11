from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from main import resolve_runtime_mode
from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.output_contracts import validate_output_directory
from src.markov_hmm_trader.runtime_modes import run_ci_smoke


def _args(**overrides):
    values = {
        "mode": None,
        "universe": False,
        "nested": False,
        "portfolio": False,
        "compare_models": False,
        "paper": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_resolve_runtime_mode_preserves_legacy_flags():
    assert resolve_runtime_mode(_args()) == "research"
    assert resolve_runtime_mode(_args(paper=True)) == "paper"
    assert resolve_runtime_mode(_args(portfolio=True)) == "portfolio"
    assert resolve_runtime_mode(_args(compare_models=True)) == "compare_models"


def test_resolve_runtime_mode_rejects_incompatible_modes():
    with pytest.raises(SystemExit, match="Incompatible runtime modes"):
        resolve_runtime_mode(_args(paper=True, mode="backtest"))
    with pytest.raises(SystemExit, match="Incompatible runtime modes"):
        resolve_runtime_mode(_args(nested=True, compare_models=True))


def test_ci_smoke_writes_required_diagnostics_contract(tmp_path):
    outputs = run_ci_smoke(
        PipelineConfig(random_seed=5, enable_experiment_tracking=False),
        output_dir=tmp_path,
        config_path=Path("configs/default.json"),
    )

    assert "stats" in outputs
    for relative in [
        "diagnostics/dependency_report.json",
        "diagnostics/dependency_report.txt",
        "diagnostics/split_registry.csv",
        "diagnostics/model_selection_candidates.csv",
        "diagnostics/baseline_comparison.csv",
        "diagnostics/experiment_metadata.json",
    ]:
        assert (tmp_path / relative).exists()
    assert all(
        result.status == "pass" for result in validate_output_directory(tmp_path, "ci_smoke")
    )
