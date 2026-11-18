"""MCP server `mcp-state-store` (Streamable HTTP, default :8081/mcp).
Tools: state_get, state_put, audit_append, audit_verify, workflow_submit, hitl_pending, hitl_decide.
hitl_* are reviewer operations and are not in any agent subagent grant."""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from reliable_assistant.errors import AssistantError  # noqa: E402
from reliable_assistant.state.store import StateStore  # noqa: E402
from reliable_assistant.types import (AuditAppendIn, AuditRecentIn, AuditRecentOut, AuditAppendOut, AuditVerifyOut, HitlDecideIn, HitlDecideOut,  # noqa: E402
                                      Meta, StateGetIn, StateGetOut, StatePutIn, StatePutOut,
                                      WorkflowSubmitIn, WorkflowSubmitOut)

PG = os.environ.get("RA_PG_DSN", "postgresql://ra_app:ra_app_dev_only@localhost:5433/ra")
REDIS = os.environ.get("RA_REDIS_URL", "redis://localhost:6390/0")
store: StateStore | None = None


@asynccontextmanager
async def lifespan(_):
    global store
    store = await StateStore.connect(PG, REDIS)
    try:
        yield
    finally:
        await store.close()


mcp = MCPServer("mcp-state-store", lifespan=lifespan)
RO = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WR = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)


@mcp.tool(annotations=RO)
async def state_get(params: StateGetIn) -> StateGetOut:
    m = params.meta
    state, ver, facts = await store.state_get(m.tenant_id, m.user_id_hash, params.session_id)
    return StateGetOut(state=state, version=ver, facts=facts)


@mcp.tool(annotations=WR)
async def state_put(params: StatePutIn) -> StatePutOut:
    m = params.meta
    if params.state.session_id is None:
        raise AssistantError(-32602, "session_id required")
    ver = await store.state_put(m.tenant_id, m.user_id_hash, params.state.session_id,
                                params.state.model_dump(mode="json"), params.expected_version,
                                [f.model_dump(mode="json") for f in params.new_facts])
    return StatePutOut(version=ver)


@mcp.tool(annotations=WR)
async def audit_append(params: AuditAppendIn) -> AuditAppendOut:
    body = params.model_dump(mode="json")
    m = body["meta"]
    body["meta"] = {k: m[k] for k in ("tenant_id", "user_id_hash", "request_id", "trace_id")}
    seq, h = await store.audit_append(m["tenant_id"], body)
    return AuditAppendOut(seq=seq, record_hash=h)


@mcp.tool(annotations=RO)
async def audit_recent(params: AuditRecentIn) -> AuditRecentOut:
    return AuditRecentOut(records=await store.audit_recent(params.meta.tenant_id, params.limit))


@mcp.tool(annotations=RO)
async def audit_verify(meta: Meta) -> AuditVerifyOut:
    ok, bad, n = await store.audit_verify(meta.tenant_id)
    return AuditVerifyOut(ok=ok, first_bad_seq=bad, records=n)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False,
                                      idempotentHint=False), description="TIER_3_MUTATE: writes PENDING_APPROVAL only; never executes.")
async def workflow_submit(params: WorkflowSubmitIn) -> WorkflowSubmitOut:
    m = params.meta
    tid = await store.workflow_submit(m.tenant_id, m.user_id_hash, params.workflow, params.payload)
    return WorkflowSubmitOut(status="PENDING_APPROVAL", hitl_ticket_id=tid)


@mcp.tool(annotations=RO)
async def hitl_pending(meta: Meta) -> list[dict]:
    return await store.hitl_pending(meta.tenant_id)


@mcp.tool(annotations=WR)
async def hitl_decide(params: HitlDecideIn) -> HitlDecideOut:
    m = params.meta
    return HitlDecideOut(status=await store.hitl_decide(m.tenant_id, m.user_id_hash, params.ticket_id, params.approve))


if __name__ == "__main__":
    import anyio
    anyio.run(lambda: mcp.run_streamable_http_async(host="127.0.0.1", port=int(os.environ.get("RA_STATE_PORT", "8081"))))
