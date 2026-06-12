from __future__ import annotations

from src.markov_hmm_trader.dependencies import (
    DEPENDENCIES,
    dependency_status,
    dependency_status_text,
)
from src.markov_hmm_trader.dependency_check import (
    check_dependencies,
    format_dependency_report,
)


def test_dependency_status_lists_required_packages():
    status = dependency_status()

    assert set(status["dependency"]) == set(DEPENDENCIES)
    assert {"group", "required"}.issubset(status.columns)
    assert set(status["status"]).issubset({"available", "missing"})


def test_dependency_status_text_contract():
    text = dependency_status_text()

    for dependency in DEPENDENCIES:
        assert f"{dependency}:" in text
    assert "[openbb_data]" in text


def test_dependency_check_wrapper_uses_stage2_contract():
    status = check_dependencies()
    text = format_dependency_report(status)

    assert "package" in status.columns
    for dependency in DEPENDENCIES:
        assert f"{dependency}:" in text


def test_dependency_check_handles_missing_package(monkeypatch):
    def fake_find_spec(module):
        return None if module == "torch" else object()

    monkeypatch.setattr(
        "src.markov_hmm_trader.dependencies.importlib.util.find_spec", fake_find_spec
    )

    status = check_dependencies()
    torch_row = status[status["package"] == "torch"].iloc[0]

    assert torch_row["status"] == "missing"
