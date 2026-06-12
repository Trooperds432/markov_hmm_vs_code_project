from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from ..config import PipelineConfig
from ..data_providers.base import BaseDataProvider
from .alt_bars import build_alt_bar_features
from .base import FeatureBlockResult
from .breadth_features import build_breadth_features
from .famafrench_features import build_famafrench_features
from .fundamental_features import build_fundamental_features
from .macro_features import build_macro_features
from .news_features import build_news_features
from .options_features import build_options_features
from .price_core import build_price_core
from .qlib_alpha_features import build_alpha158_lite
from .research_public_features import build_research_public_features
from .technical_features import build_technical_features
from .volatility_core import build_volatility_core

FeatureBuilder = Callable[
    [pd.DataFrame, PipelineConfig, BaseDataProvider | None], FeatureBlockResult
]

FEATURE_GROUP_ALIASES = {
    "qlib_alpha": "alpha158_lite",
}


def canonical_feature_group(name: str) -> str:
    return FEATURE_GROUP_ALIASES.get(name, name)


BUILDERS: dict[str, FeatureBuilder] = {
    "price_core": lambda df, cfg, provider: build_price_core(df),
    "volatility_core": lambda df, cfg, provider: build_volatility_core(df),
    "technical": lambda df, cfg, provider: build_technical_features(df),
    "alpha158_lite": lambda df, cfg, provider: build_alpha158_lite(df),
    "breadth": lambda df, cfg, provider: build_breadth_features(df, cfg, provider),
    "options": lambda df, cfg, provider: build_options_features(df, cfg, provider),
    "macro": lambda df, cfg, provider: build_macro_features(df, cfg, provider),
    "fundamentals": lambda df, cfg, provider: build_fundamental_features(df, cfg, provider),
    "news": lambda df, cfg, provider: build_news_features(df, cfg, provider),
    "famafrench": lambda df, cfg, provider: build_famafrench_features(df, cfg, provider),
    "alt_bars": lambda df, cfg, provider: build_alt_bar_features(df),
    "research_public": lambda df, cfg, provider: build_research_public_features(df, cfg, provider),
}

KNOWN_FEATURE_GROUPS = set(BUILDERS).union(FEATURE_GROUP_ALIASES)
