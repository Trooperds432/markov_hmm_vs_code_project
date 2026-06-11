from __future__ import annotations

import json

from src.markov_hmm_trader.doctor import build_dependency_report, write_doctor_reports


def test_doctor_report_writes_required_contract_keys(tmp_path):
    report = write_doctor_reports(output_dir=tmp_path)

    assert {"required", "optional", "status", "missing_required", "missing_optional"}.issubset(
        report
    )
    assert (tmp_path / "diagnostics" / "dependency_report.json").exists()
    assert (tmp_path / "diagnostics" / "dependency_report.txt").exists()
    with (tmp_path / "diagnostics" / "dependency_report.json").open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["status"] in {"pass", "warn", "fail"}


def test_doctor_optional_missing_is_warning_not_error(monkeypatch):
    def fake_package_status(distribution: str, module: str):
        if distribution == "torch":
            return {"available": False, "version": "", "module": module, "error": "missing"}
        return {"available": True, "version": "1.0", "module": module, "error": ""}

    monkeypatch.setattr("src.markov_hmm_trader.doctor._package_status", fake_package_status)
    monkeypatch.setattr("src.markov_hmm_trader.doctor._check_project_import", lambda: ("pass", ""))
    monkeypatch.setattr(
        "src.markov_hmm_trader.doctor._check_hmm_instantiation", lambda: ("pass", "")
    )
    monkeypatch.setattr("src.markov_hmm_trader.doctor._check_sklearn_core", lambda: ("pass", ""))

    report = build_dependency_report()

    assert "torch" in report["missing_optional"]
    assert report["missing_required"] == []
    assert report["status"] == "warn"
