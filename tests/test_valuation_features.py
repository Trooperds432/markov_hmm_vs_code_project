from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.markov_hmm_trader.alt_data.loaders import load_valuation_data
from src.markov_hmm_trader.alt_data.valuation import build_valuation_features
from src.markov_hmm_trader.config import PipelineConfig

FIXTURES = Path(__file__).parent / "fixtures"


def test_valuation_features_compute_margin_and_flags():
    valuation, warnings = load_valuation_data(str(FIXTURES / "valuation_sample.csv"))
    market = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2020-01-01", periods=120))
    cfg = PipelineConfig(ticker="SYNTH", valuation_publication_lag_days=3, valuation_z_window=20)

    features = build_valuation_features(market, valuation, cfg)

    assert warnings.empty
    assert {"margin_of_safety", "price_to_intrinsic", "deep_value_flag"}.issubset(features)
    assert features["margin_of_safety"].dropna().iloc[0] > 0
    assert features["price_to_intrinsic"].dropna().gt(0).all()


def test_invalid_intrinsic_value_becomes_nan():
    valuation, _ = load_valuation_data(str(FIXTURES / "valuation_sample.csv"))
    valuation.loc[valuation["ticker"].eq("SYNTH"), "graham_intrinsic_value"] = -1.0
    market = pd.DataFrame({"Close": 100.0}, index=pd.bdate_range("2020-01-01", periods=40))

    features = build_valuation_features(market, valuation, PipelineConfig(ticker="SYNTH"))

    assert features["graham_intrinsic_value"].isna().all()
