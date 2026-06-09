from __future__ import annotations

from src.markov_hmm_trader.dependencies import DEPENDENCIES, dependency_status, dependency_status_text


def test_dependency_status_lists_required_packages():
    status = dependency_status()

    assert set(status["dependency"]) == set(DEPENDENCIES)
    assert set(status["status"]).issubset({"available", "missing"})


def test_dependency_status_text_contract():
    text = dependency_status_text()

    for dependency in DEPENDENCIES:
        assert f"{dependency}:" in text
