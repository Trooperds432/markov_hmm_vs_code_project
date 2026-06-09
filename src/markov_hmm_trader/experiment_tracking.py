from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .experiment import finalize_experiment_run, start_experiment_run


def start_run(cfg: PipelineConfig, output_dir: Path) -> Path | None:
    return start_experiment_run(cfg, output_dir)


def finish_run(cfg: PipelineConfig, run_dir: Path | None, output_dir: Path, stats, selected_model):
    finalize_experiment_run(cfg, run_dir, output_dir, stats, selected_model)
