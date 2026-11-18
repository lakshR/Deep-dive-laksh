"""Per-user exact-match answer cache (ADR-3): key = {tenant}:cache:SHA256(tenant:user:normalized_query), jittered TTL
300 s ± 45 s. Only grounded answers that passed verification (faithfulness ≥ 0.92) and the post-audit are written.
NOT implemented: the semantic (cosine ≥ 0.92) tier — it needs an embedder in the API process or an `embed` MCP tool."""
import hashlib
import json
import random
import re

from .agent.entities import repair_str

MIN_FAITHFULNESS = 0.92


def norm_query(q: str) -> str:
    # key-only normalisation: upper-case first so 'lot 44' / 'LOT-44' / 'LOT-00044' collapse (the query itself is untouched)
    return re.sub(r"\s+", " ", repair_str(q.upper()).strip().lower())


def cache_key(tenant: str, user: str, query: str) -> str:
    h = hashlib.sha256(f"{tenant}:{user}:{norm_query(query)}".encode()).hexdigest()
    return f"{{{tenant}}}:cache:{h}"


def ttl_s() -> int:
    return int(300 + random.uniform(-45, 45))


class AnswerCache:
    def __init__(self, redis):
        self.redis = redis

    async def get(self, tenant: str, user: str, query: str) -> dict | None:
        try:
            raw = await self.redis.get(cache_key(tenant, user, query))
        except Exception:  # noqa: BLE001  (FMEA #1: Redis down -> bypass cache, never a shared fallback)
            return None
        return json.loads(raw) if raw else None

    async def put(self, tenant: str, user: str, query: str, value: dict, faithfulness: float | None) -> bool:
        if faithfulness is None or faithfulness < MIN_FAITHFULNESS:
            return False
        try:
            await self.redis.set(cache_key(tenant, user, query), json.dumps(value), ex=ttl_s())
            return True
        except Exception:  # noqa: BLE001
            return False
