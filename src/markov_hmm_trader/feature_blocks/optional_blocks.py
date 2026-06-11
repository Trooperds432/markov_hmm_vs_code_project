from __future__ import annotations

from ..data_providers.base import BaseDataProvider
from .base import FeatureBlockResult, skipped_feature_group

CAPABILITY_BY_GROUP = {
    "macro": "macro",
    "fundamentals": "fundamentals",
    "news": "news",
    "options": "options",
    "famafrench": "famafrench",
    "alt_bars": "alt_bars",
}


def build_optional_skip_group(name: str, provider: BaseDataProvider | None) -> FeatureBlockResult:
    capability = CAPABILITY_BY_GROUP.get(name, name)
    if provider is None:
        return skipped_feature_group(name, "provider_unavailable", capability)
    if not provider.supports(capability):
        return skipped_feature_group(name, "provider_lacks_capability", capability)
    return skipped_feature_group(name, "not_implemented_in_stage2e_foundation", capability)
