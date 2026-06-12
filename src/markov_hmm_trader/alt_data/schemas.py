from __future__ import annotations

GLASSDOOR_ALIASES = {
    "date": "review_date",
    "timestamp": "review_date",
    "date_review": "review_date",
    "symbol": "ticker",
    "overall": "overall_rating",
    "senior_management": "senior_mgmt",
    "culture": "culture_values",
    "work_life": "work_life_balance",
    "outlook": "business_outlook",
    "ceo_approval": "ceo_approv",
}

WSB_ALIASES = {
    "created_utc": "date",
    "created_dt": "date",
    "symbol": "ticker",
    "mentions": "mention_count",
    "count": "mention_count",
    "num_comments": "comment_count",
    "score": "score_sum",
}

VALUATION_ALIASES = {
    "symbol": "ticker",
    "intrinsic_value": "graham_intrinsic_value",
    "price": "close",
    "eps_ttm": "ttm_eps",
    "growth": "eps_growth_rate",
}


def canonicalize_columns(columns: list[str], aliases: dict[str, str]) -> dict[str, str]:
    return {
        column: aliases.get(column.strip().lower(), column.strip().lower()) for column in columns
    }
