from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import PipelineConfig, config_hash
from .doctor import build_dependency_report
from .model_audit import build_model_selection_candidates
from .reproducibility import experiment_metadata
from .warnings import normalize_warning_frame, warnings_to_json_records


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def git_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def create_run_folder(
    cfg: PipelineConfig,
    base_dir: Path = Path("experiments/runs"),
    timestamp: datetime | None = None,
) -> Path:
    timestamp = timestamp or datetime.now()
    run_id = f"{timestamp:%Y%m%d_%H%M%S}_{_safe_name(cfg.experiment_name)}"
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def start_experiment_run(cfg: PipelineConfig, output_dir: Path) -> Path | None:
    if not cfg.enable_experiment_tracking:
        return None

    run_dir = create_run_folder(cfg, Path(cfg.experiment_root))
    with (run_dir / "config_snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit_hash(),
        "output_dir": str(output_dir),
    }
    with (run_dir / "environment.json").open("w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2)
    return run_dir


def finalize_experiment_run(
    cfg: PipelineConfig,
    run_dir: Path | None,
    output_dir: Path,
    stats: pd.Series | None = None,
    selected_model: dict[str, object] | None = None,
) -> None:
    if run_dir is None:
        return

    if stats is not None:
        stats.to_frame("value").to_csv(run_dir / "stats.csv")
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2, default=str)

    files = []
    for path in sorted(output_dir.glob("*")):
        if path.is_file():
            files.append(path.name)
            if path.suffix.lower() in {".csv", ".json"}:
                shutil.copy2(path, run_dir / path.name)

    with (run_dir / "output_files.json").open("w", encoding="utf-8") as f:
        json.dump(files, f, indent=2)
    artifact_index = _artifact_index(output_dir)
    with (run_dir / "artifacts_index.json").open("w", encoding="utf-8") as f:
        json.dump(artifact_index, f, indent=2)
    with (output_dir / "artifacts_index.json").open("w", encoding="utf-8") as f:
        json.dump(artifact_index, f, indent=2)

    dependency_report = build_dependency_report(output_dir=output_dir)
    with (run_dir / "dependency_report.json").open("w", encoding="utf-8") as f:
        json.dump(dependency_report, f, indent=2, default=str)

    warnings_frame = (
        pd.read_csv(output_dir / "warnings.csv")
        if (output_dir / "warnings.csv").exists()
        else normalize_warning_frame(None)
    )
    with (run_dir / "warnings.json").open("w", encoding="utf-8") as f:
        json.dump(warnings_to_json_records(warnings_frame), f, indent=2, default=str)

    if (output_dir / "hmm_model_selection.csv").exists():
        model_selection = pd.read_csv(output_dir / "hmm_model_selection.csv")
        candidates = build_model_selection_candidates(model_selection, cfg.ticker)
        candidates.to_csv(run_dir / "model_selection_candidates.csv", index=False)
        candidates.to_csv(output_dir / "model_selection_candidates.csv", index=False)

    manifest = {
        "run_id": run_dir.name,
        "config": asdict(cfg),
        "selected_models": selected_model or {},
        "file_inventory": files,
        "artifact_count": len(artifact_index),
        "warning_count": int(len(warnings_frame)),
        "provider": cfg.data_provider,
        "feature_groups_enabled": list(cfg.feature_groups_enabled),
        "skipped_optional_groups": sorted(
            set(warnings_frame.get("feature_group", pd.Series(dtype=str)).dropna())
        ),
        "git_commit": git_commit_hash(),
    }
    with (run_dir / "experiment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    with (output_dir / "experiment_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    with (output_dir / "experiment_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(experiment_metadata(cfg), f, indent=2, default=str)

    append_model_registry(cfg, run_dir, stats, selected_model or {})


def _artifact_index(output_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(output_dir.glob("*")):
        if not path.is_file():
            continue
        rows.append(
            {
                "path": path.name,
                "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )
    return rows


def append_model_registry(
    cfg: PipelineConfig,
    run_dir: Path,
    stats: pd.Series | None,
    selected_model: dict[str, object],
    approved_for_paper_trading: bool = False,
    notes: str = "",
) -> None:
    registry_path = Path(cfg.model_registry_path)
    registry_dir = registry_path.parent
    registry_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": run_dir.name,
        "timestamp": run_dir.name.split("_", 2)[0],
        "ticker": cfg.ticker,
        "config_hash": config_hash(cfg),
        "model_type": selected_model.get("model_type", "GaussianHMM"),
        "selected_n_states": selected_model.get("selected_n_states", cfg.n_states),
        "covariance_type": selected_model.get("covariance_type", cfg.covariance_type),
        "validation_score": selected_model.get("validation_score", ""),
        "oos_cagr": "" if stats is None else stats.get("strategy_cagr", ""),
        "oos_sharpe": "" if stats is None else stats.get("strategy_sharpe", ""),
        "oos_max_drawdown": "" if stats is None else stats.get("strategy_max_drawdown", ""),
        "approved_for_paper_trading": bool(approved_for_paper_trading),
        "notes": notes,
    }
    existing = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(registry_path, index=False)
