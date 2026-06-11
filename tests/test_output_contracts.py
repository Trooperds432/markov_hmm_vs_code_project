from __future__ import annotations

import json

import pandas as pd

from src.markov_hmm_trader.output_contracts import (
    CsvContract,
    JsonContract,
    validate_csv_contract,
    validate_json_contract,
    validate_output_directory,
)


def test_csv_contract_passes_and_reports_missing_columns(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame([{"a": 1, "b": 2}]).to_csv(path, index=False)

    result = validate_csv_contract(path, CsvContract("sample.csv", ("a", "b"), min_rows=1))
    failure = validate_csv_contract(path, CsvContract("sample.csv", ("missing",), min_rows=1))

    assert result.status == "pass"
    assert failure.status == "fail"
    assert "missing_columns" in failure.message


def test_json_contract_passes_and_reports_missing_keys(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    result = validate_json_contract(path, JsonContract("sample.json", ("a",)))
    failure = validate_json_contract(path, JsonContract("sample.json", ("b",)))

    assert result.status == "pass"
    assert failure.status == "fail"
    assert "missing_keys" in failure.message


def test_stage2d_output_directory_contracts(tmp_path):
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "dependency_report.json").write_text(
        json.dumps(
            {
                "required": {},
                "optional": {},
                "status": "pass",
                "missing_required": [],
                "missing_optional": [],
            }
        ),
        encoding="utf-8",
    )
    (diagnostics / "experiment_metadata.json").write_text(
        json.dumps(
            {
                "random_seed": 42,
                "deterministic": True,
                "config_hash": "abc",
                "git_commit": "def",
                "created_at": "2020-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "split_id": "x",
                "train_start": "a",
                "train_end": "b",
                "test_start": "c",
                "test_end": "d",
                "purge_bars": 0,
                "embargo_bars": 0,
            }
        ]
    ).to_csv(diagnostics / "split_registry.csv", index=False)
    pd.DataFrame(
        [{"model_family": "hmm", "candidate_id": 1, "score": 1.0, "selected": True}]
    ).to_csv(diagnostics / "model_selection_candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "baseline": "cash",
                "total_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "final_equity": 1.0,
            }
        ]
    ).to_csv(diagnostics / "baseline_comparison.csv", index=False)

    results = validate_output_directory(tmp_path, mode="ci_smoke")

    assert all(result.status == "pass" for result in results)
