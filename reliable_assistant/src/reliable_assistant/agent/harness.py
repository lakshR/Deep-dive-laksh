"""ToolExecutionHarness — every tool call passes: allow-list -> strict validation -> auto-repair+revalidate
-> RBAC tier -> UUIDv4 idempotency -> TOOL_DISPATCH plugins (circuit_breaker, hitl) -> execute (§1.2.6)."""
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Awaitable, Callable
from uuid import UUID

from pydantic import BaseModel, ValidationError

from ..errors import AssistantError
from ..plugins.manager import PluginManager
from .entities import repair_value

TIER_RANK = {"TIER_1_READ": 1, "TIER_2_COMPUTE": 2, "TIER_3_MUTATE": 3}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    tier: str
    in_model: type[BaseModel]
    handler: Callable[[BaseModel], Awaitable[dict]]


class MemoryIdempotency:
    """Default (tests). Production uses RedisIdempotency (shared across pods)."""

    def __init__(self):
        self._d: dict[str, dict] = {}

    async def get(self, tenant: str, key: str) -> dict | None:
        return self._d.get(f"{tenant}:{key}")

    async def set(self, tenant: str, key: str, value: dict) -> None:
        self._d[f"{tenant}:{key}"] = value


class RedisIdempotency:
    TTL_S = 24 * 3600

    def __init__(self, redis):
        self.redis = redis

    async def get(self, tenant: str, key: str) -> dict | None:
        raw = await self.redis.get(f"{{{tenant}}}:idem:{key}")
        return json.loads(raw) if raw else None

    async def set(self, tenant: str, key: str, value: dict) -> None:
        await self.redis.set(f"{{{tenant}}}:idem:{key}", json.dumps(value, default=str), ex=self.TTL_S)


@dataclass(frozen=True)
class Grant:  # per-subagent RBAC (§1.2.7)
    subagent: str
    tools: frozenset[str]
    max_tier: str


def args_hash(args: dict) -> str:
    """Canonical hash excluding `meta` (which differs on every call)."""
    body = {k: v for k, v in args.items() if k != "meta"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


class ToolExecutionHarness:
    def __init__(self, specs: list[ToolSpec], plugins: PluginManager, idempotency=None):
        self.specs = {s.name: s for s in specs}
        self.plugins = plugins
        self.metrics: Counter = Counter()   # KPI-14 / KPI-15 telemetry
        self.audit: list[dict] = []         # originals preserved when repaired
        self.idem = idempotency or MemoryIdempotency()

    async def call(self, tool: str, args: dict, ctx: dict, grant: Grant) -> dict:
        spec = self.specs.get(tool)
        if spec is None or tool not in grant.tools:                       # (d) tenant/subagent allow-list
            raise AssistantError(-32601, f"tool {tool!r} not available")
        model = self._validate(spec, args)
        if TIER_RANK[spec.tier] > TIER_RANK[grant.max_tier]:               # RBAC tier check
            raise AssistantError(-32012, f"{grant.subagent} not granted {spec.tier}")
        if model.meta.tenant_id != ctx["tenant_id"]:
            raise AssistantError(-32011, "meta.tenant_id mismatch")
        key = model.meta.idempotency_key
        if not isinstance(key, UUID) or key.version != 4:
            raise AssistantError(-32602, "idempotency_key must be UUIDv4")
        tier3 = spec.tier == "TIER_3_MUTATE"
        stored = await self.idem.get(ctx["tenant_id"], str(key))
        if stored is not None:                                             # duplicate -> never re-execute
            self.metrics["idempotent_replays"] += 1
            if tier3:
                raise AssistantError(-32015, "awaiting human approval", stored)
            return stored
        call = {"tool": tool, "tier": spec.tier, "subagent": grant.subagent,
                "args_hash": args_hash(model.model_dump(mode="json"))}   # canonical: AFTER repair, so SKU99 == SKU-099
        try:
            self.plugins.run_verdicts("TOOL_DISPATCH", ctx=ctx, call=call)   # circuit_breaker -> hitl
        except AssistantError as e:
            if not (e.code == -32015 and tier3):
                raise
        if tier3:
            # Tier 3 is never executed. The handler ONLY records PENDING_APPROVAL (mcp-state-store); if that write
            # fails the call fails closed (-32002) and no action is possible. Enforced here even if `hitl` is disabled.
            pending = await spec.handler(model)
            await self.idem.set(ctx["tenant_id"], str(key), pending)
            raise AssistantError(-32015, "awaiting human approval", pending)
        result = await spec.handler(model)
        await self.idem.set(ctx["tenant_id"], str(key), result)
        return result

    def _validate(self, spec: ToolSpec, args: dict) -> BaseModel:
        try:
            m = spec.in_model.model_validate(args)
            self.metrics["first_pass_ok"] += 1
            return m
        except ValidationError as first:
            self.metrics["first_pass_fail"] += 1
            fixed = repair_value(args)
            if fixed != args:
                try:
                    m = spec.in_model.model_validate(fixed)
                    self.metrics["repair_ok"] += 1
                    self.audit.append({"tool": spec.name, "original": args, "repaired": fixed})
                    return m
                except ValidationError:
                    pass
            self.metrics["repair_fail"] += 1
            raise AssistantError(-32602, str(first.errors(include_url=False))[:300]) from first
