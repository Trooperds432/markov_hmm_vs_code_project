from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

ContractStatus = Literal["pass", "warn", "fail"]


@dataclass(frozen=True)
class CsvContract:
    name: str
    required_columns: tuple[str, ...]
    min_rows: int = 0
    required: bool = True


@dataclass(frozen=True)
class JsonContract:
    name: str
    required_keys: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class ContractResult:
    name: str
    status: ContractStatus
    message: str
    path: str


STAGE2D_CSV_CONTRACTS: tuple[CsvContract, ...] = (
    CsvContract(
        "diagnostics/split_registry.csv",
        (
            "split_id",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "purge_bars",
            "embargo_bars",
        ),
        min_rows=1,
    ),
    CsvContract(
        "diagnostics/model_selection_candidates.csv",
        ("model_family", "candidate_id", "score", "selected"),
        min_rows=1,
    ),
    CsvContract(
        "diagnostics/baseline_comparison.csv",
        ("baseline", "total_return", "sharpe", "max_drawdown", "final_equity"),
        min_rows=1,
    ),
)

STAGE2D_JSON_CONTRACTS: tuple[JsonContract, ...] = (
    JsonContract(
        "diagnostics/dependency_report.json",
        ("required", "optional", "status", "missing_required", "missing_optional"),
    ),
    JsonContract(
        "diagnostics/experiment_metadata.json",
        ("random_seed", "deterministic", "config_hash", "git_commit", "created_at"),
    ),
)


def validate_csv_contract(path: Path, contract: CsvContract) -> ContractResult:
    if not path.exists():
        status: ContractStatus = "fail" if contract.required else "warn"
        return ContractResult(contract.name, status, "missing_file", str(path))
    try:
        frame = pd.read_csv(path)
    except Exception as err:
        return ContractResult(
            contract.name, "fail", f"csv_read_error:{type(err).__name__}", str(path)
        )

    missing = [column for column in contract.required_columns if column not in frame.columns]
    if missing:
        return ContractResult(
            contract.name,
            "fail",
            f"missing_columns:{','.join(missing)}",
            str(path),
        )
    if len(frame) < contract.min_rows:
        return ContractResult(
            contract.name,
            "fail",
            f"insufficient_rows:{len(frame)}<{contract.min_rows}",
            str(path),
        )
    return ContractResult(contract.name, "pass", "ok", str(path))


def validate_json_contract(path: Path, contract: JsonContract) -> ContractResult:
    if not path.exists():
        status: ContractStatus = "fail" if contract.required else "warn"
        return ContractResult(contract.name, status, "missing_file", str(path))
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as err:
        return ContractResult(
            contract.name,
            "fail",
            f"json_read_error:{type(err).__name__}",
            str(path),
        )
    missing = [key for key in contract.required_keys if key not in payload]
    if missing:
        return ContractResult(contract.name, "fail", f"missing_keys:{','.join(missing)}", str(path))
    return ContractResult(contract.name, "pass", "ok", str(path))


def validate_output_directory(output_dir: Path, mode: str = "research") -> list[ContractResult]:
    results: list[ContractResult] = []
    csv_contracts = STAGE2D_CSV_CONTRACTS if mode in {"ci_smoke", "diagnostics"} else ()
    json_contracts = STAGE2D_JSON_CONTRACTS if mode in {"ci_smoke", "diagnostics"} else ()
    for contract in csv_contracts:
        results.append(validate_csv_contract(output_dir / contract.name, contract))
    for contract in json_contracts:
        results.append(validate_json_contract(output_dir / contract.name, contract))
    return results


def assert_contract_results(results: list[ContractResult]) -> None:
    failures = [result for result in results if result.status == "fail"]
    if failures:
        message = "; ".join(f"{result.name}:{result.message}" for result in failures)
        raise ValueError(f"Output contract validation failed: {message}")
