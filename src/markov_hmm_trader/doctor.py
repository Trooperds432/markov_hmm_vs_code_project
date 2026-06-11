from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any

from .config import load_config
from .reproducibility import get_git_sha, get_working_tree_dirty, utc_now_iso

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scikit-learn": "sklearn",
    "hmmlearn": "hmmlearn",
    "yfinance": "yfinance",
    "statsmodels": "statsmodels",
    "pytest": "pytest",
    "ruff": "ruff",
    "black": "black",
}

OPTIONAL_PACKAGES = {
    "torch": "torch",
    "arch": "arch",
    "statsforecast": "statsforecast",
    "openbb": "openbb",
}


def _package_status(distribution: str, module: str) -> dict[str, Any]:
    try:
        imported = importlib.import_module(module)
        available = True
        error = ""
    except Exception as err:
        imported = None
        available = False
        error = type(err).__name__
    version = ""
    if available:
        try:
            version = importlib.metadata.version(distribution)
        except Exception:
            version = getattr(imported, "__version__", "")
    return {"available": available, "version": version, "module": module, "error": error}


def _check_project_import() -> tuple[str, str]:
    try:
        import src.markov_hmm_trader

        return "pass", str(src.markov_hmm_trader.__file__)
    except Exception as err:
        return "fail", type(err).__name__


def _check_hmm_instantiation() -> tuple[str, str]:
    try:
        from hmmlearn.hmm import GaussianHMM

        GaussianHMM(n_components=2, covariance_type="diag", n_iter=1, random_state=1)
        return "pass", "GaussianHMM"
    except Exception as err:
        return "fail", type(err).__name__


def _check_sklearn_core() -> tuple[str, str]:
    try:
        from sklearn.preprocessing import StandardScaler

        StandardScaler()
        return "pass", "StandardScaler"
    except Exception as err:
        return "fail", type(err).__name__


def _output_dir_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def build_dependency_report(
    config_path: Path = Path("configs/default.json"),
    output_dir: Path = Path("outputs"),
) -> dict[str, Any]:
    repo_root = Path.cwd()
    diagnostics_dir = output_dir / "diagnostics"
    required = {name: _package_status(name, module) for name, module in REQUIRED_PACKAGES.items()}
    optional = {name: _package_status(name, module) for name, module in OPTIONAL_PACKAGES.items()}
    missing_required = [name for name, info in required.items() if not info["available"]]
    missing_optional = [name for name, info in optional.items() if not info["available"]]
    warnings = [f"optional_missing:{name}" for name in missing_optional]
    errors = [f"required_missing:{name}" for name in missing_required]

    project_import_status, project_import_detail = _check_project_import()
    hmm_status, hmm_detail = _check_hmm_instantiation()
    sklearn_status, sklearn_detail = _check_sklearn_core()
    config_status = "pass"
    config_detail = str(config_path)
    try:
        load_config(config_path)
    except Exception as err:
        config_status = "fail"
        config_detail = type(err).__name__

    checks = {
        "project_import": project_import_status,
        "project_import_detail": project_import_detail,
        "config_load": config_status,
        "config_load_detail": config_detail,
        "hmm_instantiation": hmm_status,
        "hmm_instantiation_detail": hmm_detail,
        "sklearn_core": sklearn_status,
        "sklearn_core_detail": sklearn_detail,
        "yfinance_import": "pass" if required["yfinance"]["available"] else "fail",
    }
    for name, status in checks.items():
        if name.endswith("_detail"):
            continue
        if status == "fail":
            errors.append(f"check_failed:{name}")

    output_writable = _output_dir_writable(diagnostics_dir)
    if not output_writable:
        errors.append("output_dir_not_writable")

    status = "fail" if errors else "warn" if warnings else "pass"
    return {
        "status": status,
        "timestamp_utc": utc_now_iso(),
        "required": required,
        "optional": optional,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "git": {
            "sha": get_git_sha(),
            "dirty": get_working_tree_dirty(),
        },
        "paths": {
            "repo_root": str(repo_root),
            "data_dir_exists": (repo_root / "data").exists(),
            "output_dir_writable": output_writable,
        },
        "packages": {**required, **optional},
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def format_dependency_report_text(report: dict[str, Any]) -> str:
    lines = [
        f"Stage 2D Doctor: {report['status'].upper()}",
        f"Timestamp UTC: {report['timestamp_utc']}",
        f"Python: {report['python']['version'].split()[0]}",
        f"Git SHA: {report['git']['sha'] or 'unavailable'}",
        f"Git dirty: {report['git']['dirty']}",
        "",
        "Required packages:",
    ]
    for name, info in report["required"].items():
        status = "available" if info["available"] else "missing"
        version = f" {info['version']}" if info["version"] else ""
        lines.append(f"- {name}: {status}{version}")
    lines.append("")
    lines.append("Optional packages:")
    for name, info in report["optional"].items():
        status = "available" if info["available"] else "missing"
        version = f" {info['version']}" if info["version"] else ""
        lines.append(f"- {name}: {status}{version}")
    if report["warnings"]:
        lines.extend(["", "Warnings:", *[f"- {warning}" for warning in report["warnings"]]])
    if report["errors"]:
        lines.extend(["", "Errors:", *[f"- {error}" for error in report["errors"]]])
    return "\n".join(lines)


def write_doctor_reports(
    config_path: Path = Path("configs/default.json"),
    output_dir: Path = Path("outputs"),
) -> dict[str, Any]:
    report = build_dependency_report(config_path=config_path, output_dir=output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    with (diagnostics_dir / "dependency_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    (diagnostics_dir / "dependency_report.txt").write_text(
        format_dependency_report_text(report),
        encoding="utf-8",
    )
    return report
