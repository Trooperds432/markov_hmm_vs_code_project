from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class FeatureBlockResult:
    name: str
    features: pd.DataFrame
    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: pd.DataFrame = field(default_factory=pd.DataFrame)
    produced_columns: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)


def feature_metadata(name: str, columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"feature_group": name, "feature": column, "status": "produced"} for column in columns]
    )


def skipped_feature_group(name: str, reason: str, detail: str = "") -> FeatureBlockResult:
    warning = pd.DataFrame(
        [
            {
                "feature_group": name,
                "feature": "",
                "reason": reason,
                "detail": detail,
            }
        ]
    )
    return FeatureBlockResult(
        name=name,
        features=pd.DataFrame(),
        warnings=warning,
        skipped=[{"feature_group": name, "reason": reason, "detail": detail}],
    )
