from __future__ import annotations

import pandas as pd

from src.markov_hmm_trader.config import PipelineConfig
from src.markov_hmm_trader.data import data_provider_warnings, load_market_data
from src.markov_hmm_trader.data_providers.csv_provider import CSVProvider
from src.markov_hmm_trader.data_providers.factory import get_data_provider
from src.markov_hmm_trader.data_providers.openbb_provider import OpenBBProvider
from src.markov_hmm_trader.data_providers.synthetic_provider import SyntheticProvider
from src.markov_hmm_trader.data_providers.yfinance_provider import YFinanceProvider


def _ohlcv_frame(periods: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-01", periods=periods)
    close = pd.Series([100.0 + i for i in range(periods)], index=dates)
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=dates,
    )


def test_synthetic_provider_is_deterministic_and_supports_breadth():
    cfg = PipelineConfig(data_provider="synthetic", synthetic_periods=120, random_seed=11)
    provider = SyntheticProvider(cfg)

    first = provider.get_price_history("AAA", cfg.start, cfg.end, cfg.interval).data
    second = provider.get_price_history("AAA", cfg.start, cfg.end, cfg.interval).data
    breadth = provider.get_market_snapshots(["AAA", "BBB"]).data

    assert provider.supports("price_history")
    assert provider.supports("breadth")
    pd.testing.assert_frame_equal(first, second)
    assert {"return_20d", "vol_20d"}.issubset(breadth.columns)


def test_csv_provider_loads_price_history(tmp_path):
    path = tmp_path / "prices.csv"
    _ohlcv_frame().rename_axis("Date").to_csv(path)
    cfg = PipelineConfig(
        data_provider="csv", csv_price_path=str(path), csv_benchmark_path=str(path)
    )

    result = CSVProvider(cfg).get_price_history("AAA", cfg.start, cfg.end, cfg.interval)

    assert not result.data.empty
    assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(result.data.columns)
    assert result.metadata.iloc[0]["provider"] == "csv"


def test_yfinance_provider_uses_mocked_download(monkeypatch, tmp_path):
    import yfinance as yf

    def fake_download(*args, **kwargs):
        return _ohlcv_frame()

    monkeypatch.setattr(yf, "download", fake_download)
    cfg = PipelineConfig(data_cache_dir=str(tmp_path), data_failover_to_cache=True)

    result = YFinanceProvider(cfg).get_price_history("AAA", cfg.start, cfg.end, cfg.interval)

    assert not result.data.empty
    assert (tmp_path / "AAA.csv").exists()


def test_openbb_missing_dependency_returns_structured_warning(monkeypatch):
    monkeypatch.setattr(
        "src.markov_hmm_trader.data_providers.openbb_provider.importlib.util.find_spec",
        lambda name: None,
    )
    cfg = PipelineConfig(data_provider="openbb")
    provider = OpenBBProvider(cfg)

    result = provider.get_price_history("AAA", cfg.start, cfg.end, cfg.interval)

    assert not provider.supports("price_history")
    assert result.data.empty
    assert result.warnings.iloc[0]["reason"] == "missing_optional_dependency"


def test_load_market_data_with_synthetic_provider_preserves_warning_contract():
    cfg = PipelineConfig(data_provider="synthetic", synthetic_periods=120)

    data = load_market_data(cfg)
    warnings = data_provider_warnings(data)

    assert not data.empty
    assert "Benchmark_Close" in data
    assert set(["provider", "symbol", "capability", "reason", "detail"]).issubset(warnings.columns)


def test_provider_factory_rejects_unknown_provider():
    cfg = PipelineConfig(data_provider="not_real")

    try:
        get_data_provider(cfg)
    except ValueError as err:
        assert "Unknown data_provider" in str(err)
    else:
        raise AssertionError("expected unknown provider to fail")
