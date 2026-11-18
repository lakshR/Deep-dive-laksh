"""StateStore: Postgres (RLS, hash-chained audit, HITL tickets, session_facts) + Redis (hot session window).
Fronted by the mcp-state-store server. Every Redis key and SQL statement is tenant-scoped."""
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import redis.asyncio as aioredis
from ulid import ULID

from ..errors import AssistantError

GENESIS = "0" * 64
SCHEMA = Path(__file__).with_name("schema.sql")
SESSION_TTL_S = 24 * 3600

# Compare-and-set on the session version; returns new version or -1 on mismatch.
_CAS = """
local cur = redis.call('HGET', KEYS[1], 'ver')
if (not cur and tonumber(ARGV[1]) ~= 0) or (cur and tonumber(cur) ~= tonumber(ARGV[1])) then return -1 end
local nv = tonumber(ARGV[1]) + 1
redis.call('HSET', KEYS[1], 'ver', nv, 'state', ARGV[2])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[3]))
return nv
"""


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def chain_hash(prev_hash: str, payload: dict) -> str:
    return hashlib.sha256((prev_hash + canonical(payload)).encode()).hexdigest()


async def apply_schema(admin_dsn: str) -> None:
    conn = await asyncpg.connect(admin_dsn)
    try:
        await conn.execute(SCHEMA.read_text())
    finally:
        await conn.close()


class StateStore:
    def __init__(self, pool: asyncpg.Pool, redis: aioredis.Redis):
        self.pool, self.redis = pool, redis

    @classmethod
    async def connect(cls, pg_dsn: str, redis_url: str) -> "StateStore":
        return cls(await asyncpg.create_pool(pg_dsn, min_size=1, max_size=16), aioredis.from_url(redis_url))

    async def close(self):
        await self.pool.close()
        await self.redis.aclose()

    @asynccontextmanager
    async def tx(self, tenant_id: str):
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.execute("SELECT set_config('app.tenant_id', $1, true)", tenant_id)  # == SET LOCAL
            yield conn

    # ---- session (Redis hot window; key hash-tagged by tenant, scoped to user) ----
    @staticmethod
    def _skey(tenant: str, user: str, session: str) -> str:
        return f"{{{tenant}}}:sess:{user}:{session}"

    async def state_get(self, tenant: str, user: str, session: str) -> tuple[dict | None, int, list[dict]]:
        h = await self.redis.hgetall(self._skey(tenant, user, session))
        state = json.loads(h[b"state"]) if h else None
        ver = int(h[b"ver"]) if h else 0
        async with self.tx(tenant) as c:
            rows = await c.fetch("SELECT fact, source_chunk_id FROM session_facts WHERE tenant_id=$1 AND "
                                 "session_id=$2 AND user_hash=$3 ORDER BY id DESC LIMIT 20", tenant, session, user)
        return state, ver, [dict(r) for r in rows]

    async def state_put(self, tenant: str, user: str, session: str, state: dict, expected_version: int,
                        facts: list[dict]) -> int:
        nv = int(await self.redis.eval(_CAS, 1, self._skey(tenant, user, session), expected_version,
                                       canonical(state), SESSION_TTL_S))
        if nv < 0:
            raise AssistantError(-32021, "session version conflict")
        if facts:
            async with self.tx(tenant) as c:
                await c.executemany("INSERT INTO session_facts (tenant_id, session_id, user_hash, fact, source_chunk_id)"
                                    " VALUES ($1,$2,$3,$4,$5)",
                                    [(tenant, session, user, f["fact"], f["source_chunk_id"]) for f in facts])
        return nv

    # ---- audit: per-tenant SHA-256 hash chain, append-only ----
    async def audit_append(self, tenant: str, payload: dict) -> tuple[int, str]:
        async with self.tx(tenant) as c:
            await c.execute("SELECT pg_advisory_xact_lock(hashtext($1))", tenant)  # serialize chain head per tenant
            last = await c.fetchrow("SELECT seq, record_hash FROM audit_log WHERE tenant_id=$1 "
                                    "ORDER BY seq DESC LIMIT 1", tenant)
            seq, prev = (last["seq"] + 1, last["record_hash"]) if last else (1, GENESIS)
            h = chain_hash(prev, payload)
            await c.execute("INSERT INTO audit_log (tenant_id, seq, prev_hash, record_hash, payload) "
                            "VALUES ($1,$2,$3,$4,$5::jsonb)", tenant, seq, prev, h, canonical(payload))
        return seq, h

    async def audit_recent(self, tenant: str, limit: int) -> list[dict]:
        async with self.tx(tenant) as c:
            rows = await c.fetch("SELECT seq, record_hash, payload, created_at FROM audit_log WHERE tenant_id=$1 "
                                 "ORDER BY seq DESC LIMIT $2", tenant, limit)
        return [{"seq": r["seq"], "record_hash": r["record_hash"], "created_at": r["created_at"].isoformat(),
                 "payload": json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]} for r in rows]

    async def audit_verify(self, tenant: str) -> tuple[bool, int | None, int]:
        async with self.tx(tenant) as c:
            rows = await c.fetch("SELECT seq, prev_hash, record_hash, payload FROM audit_log WHERE tenant_id=$1 "
                                 "ORDER BY seq", tenant)
        prev = GENESIS
        for r in rows:
            payload = json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
            if r["prev_hash"] != prev or r["record_hash"] != chain_hash(prev, payload):
                return False, r["seq"], len(rows)
            prev = r["record_hash"]
        return True, None, len(rows)

    # ---- HITL (Tier 3): always PENDING_APPROVAL, never executed ----
    async def workflow_submit(self, tenant: str, user: str, workflow: str, payload: dict) -> str:
        tid = f"HITL-{ULID()}"
        async with self.tx(tenant) as c:
            await c.execute("INSERT INTO hitl_tickets (ticket_id, tenant_id, user_hash, workflow, payload) "
                            "VALUES ($1,$2,$3,$4,$5::jsonb)", tid, tenant, user, workflow, canonical(payload))
        return tid

    async def hitl_pending(self, tenant: str) -> list[dict]:
        async with self.tx(tenant) as c:
            rows = await c.fetch("SELECT ticket_id, user_hash, workflow, payload, requested_at FROM hitl_tickets "
                                 "WHERE tenant_id=$1 AND status='PENDING_APPROVAL' ORDER BY requested_at", tenant)
        return [{**dict(r), "payload": json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"],
                 "requested_at": r["requested_at"].isoformat()} for r in rows]

    async def hitl_decide(self, tenant: str, reviewer: str, ticket_id: str, approve: bool) -> str:
        async with self.tx(tenant) as c:
            row = await c.fetchrow("SELECT user_hash, status FROM hitl_tickets WHERE ticket_id=$1 AND tenant_id=$2 "
                                   "FOR UPDATE", ticket_id, tenant)
            if row is None:
                raise AssistantError(-32012, "ticket not found in this tenant")
            if row["status"] != "PENDING_APPROVAL":
                raise AssistantError(-32021, f"ticket already {row['status']}")
            if row["user_hash"] == reviewer:
                raise AssistantError(-32012, "requester cannot approve their own request")
            status = "APPROVED" if approve else "REJECTED"
            await c.execute("UPDATE hitl_tickets SET status=$1, decided_by=$2, decided_at=now() WHERE ticket_id=$3 "
                            "AND tenant_id=$4", status, reviewer, ticket_id, tenant)
        return status
