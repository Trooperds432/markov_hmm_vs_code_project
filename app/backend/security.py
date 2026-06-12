from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from . import settings


def password_required() -> bool:
    return bool(settings.APP_PASSWORD)


def check_password(candidate: str | None) -> bool:
    if not password_required():
        return True
    if not candidate:
        return False
    return hmac.compare_digest(candidate, settings.APP_PASSWORD)


async def require_auth(x_app_password: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing the optional APP_PASSWORD gate."""
    if not password_required():
        return
    if not check_password(x_app_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing app password.",
        )
