"""JWT -> (tenant_id, user_hash). JWT claim is the only source of truth for tenant (§1.2.8)."""
import hashlib
import hmac
import os
import re

import jwt

from .errors import AssistantError

TENANT_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")


def settings() -> dict:
    # Dev default only; production uses OIDC/JWKS (RS256). Refuse the default outside dev.
    return {"secret": os.environ.get("RA_JWT_SECRET", "dev-only-secret-change-me-32-bytes!!"),
            "hmac_key": os.environ.get("RA_USER_HMAC_KEY", "dev-only-hmac-key").encode()}


def user_hash(user_id: str) -> str:
    return hmac.new(settings()["hmac_key"], user_id.encode(), hashlib.sha256).hexdigest()[:32]


def mint_token(tenant_id: str, user_id: str, role: str = "user") -> str:  # for tests / load-test tenants
    return jwt.encode({"tenant_id": tenant_id, "sub": user_id, "role": role}, settings()["secret"], algorithm="HS256")


def authenticate(authorization: str | None, asserted_tenant: str | None = None,
                 header_tenant: str | None = None, with_role: bool = False):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AssistantError(-32011, "missing bearer token")
    try:
        claims = jwt.decode(authorization[7:], settings()["secret"], algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise AssistantError(-32011, f"invalid token: {exc.__class__.__name__}") from exc
    tenant, sub = claims.get("tenant_id"), claims.get("sub")
    if not isinstance(tenant, str) or not TENANT_RE.match(tenant) or not sub:
        raise AssistantError(-32011, "token lacks valid tenant_id/sub")
    for asserted in (asserted_tenant, header_tenant):
        if asserted is not None and asserted != tenant:
            raise AssistantError(-32011, "tenant assertion does not match token")
    uh = user_hash(str(sub))
    return (tenant, uh, claims.get("role", "user")) if with_role else (tenant, uh)
