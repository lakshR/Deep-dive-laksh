"""FastAPI app — design.md §1.2.8. Stage 4: auth -> PII/injection pre-hooks -> session -> triage -> bounded loop
-> egress -> hash-chained audit (WAL fallback). Tier 3 -> PENDING_APPROVAL via mcp-state-store, reviewer API."""
import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import redis.asyncio as aioredis
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
from ulid import ULID

from .agent.entities import extract_entities
from .agent.harness import RedisIdempotency, ToolExecutionHarness
from .agent.loop import DEV_BUDGET, LoopBudget, ReActLoop, State
from .agent.tools import build_specs
from .agent.triage import triage as run_triage
from .auth import authenticate
from .errors import HTTP_STATUS, AssistantError
from .llm.client import GEN_MODEL, VERIFY_MODEL, make_client
from .llm.ground import generate_grounded, linked_as_chunks
from .cache import AnswerCache
from .mcpclient import McpClient
from .quality import audit_answer  # noqa: F401  (re-exported for tests)
from .telemetry import Recorder
from .plugins.builtin import DEFAULTS
from .plugins.manager import PluginManager
from .plugins.pii import redact
from .agent.compress import approx_tokens
from .types import (AgentQueryIn, AgentQueryOut, Citation, GoalAnchorBlock, SessionFact, SessionState,
                    RerankOut, StateGetOut, StatePutOut, AuditAppendOut, AuditRecentOut)

log = logging.getLogger("reliable_assistant")
ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(os.environ.get("RA_MCP_CONFIG", ROOT / "mcp_config.json"))
STATE_URL = os.environ.get("RA_STATE_URL", "http://localhost:8081/mcp")
REDIS_URL = os.environ.get("RA_REDIS_URL", "redis://localhost:6390/0")
WAL_PATH = Path(os.environ.get("RA_AUDIT_WAL", ROOT / "var" / "audit_wal.jsonl"))
WAL_MAX = 10_000                                     # FMEA #6: beyond this, high-risk domains fail closed
HIGH_RISK = {"clinical", "financial", "legal"}

ABSTAIN = "I can't answer that from the knowledge available to your organization."
REFUSAL = "That request is outside what this assistant can help with."
GREETING = "Hello! Ask me a question about your organization's documents and procedures."
# Placeholder wording — requires legal sign-off (design.md Appendix B item 15).
DISCLAIMERS = {"clinical": "Informational only, not clinical advice. ", "financial": "Informational only, not financial advice. ",
               "legal": "Informational only, not legal advice. "}
POLICY_CONSTRAINTS = ["answer only from retrieved evidence", "never execute state-changing actions without human approval"]


def build_plugins() -> PluginManager:
    pm = PluginManager.from_file(CONFIG)
    for name in pm.config:
        if name in DEFAULTS:
            pm.register(name, DEFAULTS[name](pm.config[name]))
    return pm


def budget_from_env() -> LoopBudget:
    return LoopBudget() if os.environ.get("RA_LOOP_PROFILE", "dev") == "spec" else DEV_BUDGET


def make_meta(ctx: dict) -> dict:
    return {"tenant_id": ctx["tenant_id"], "user_id_hash": ctx["user_id_hash"], "request_id": ctx["request_id"],
            "trace_id": ctx["trace_id"], "deadline_ms": 2000, "idempotency_key": str(uuid.uuid4())}


class Decision(BaseModel):
    decision: str  # "approve" | "reject"


# ---- audit with local WAL fallback (FMEA #6) ----
def _wal_lines() -> list[str]:
    return WAL_PATH.read_text().splitlines() if WAL_PATH.exists() else []


async def audit(app: FastAPI, params: dict, domain: str = "general") -> None:
    state: McpClient | None = app.state.state
    try:
        if state is None:
            raise AssistantError(-32002, "state store unavailable")
        pending = _wal_lines()
        if pending:                                           # replay in original order before the new record
            for line in pending:
                await state.call("audit_append", json.loads(line), AuditAppendOut)
            WAL_PATH.unlink()
        await state.call("audit_append", params, AuditAppendOut)
    except AssistantError as exc:
        n = len(_wal_lines())
        if n >= WAL_MAX and domain in HIGH_RISK:
            raise AssistantError(-32002, "audit WAL full; failing closed for high-risk domain") from exc
        WAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with WAL_PATH.open("a") as f:
            f.write(json.dumps(params) + "\n")
        log.warning("audit written to WAL (%d pending): %s", n + 1, exc)


def _record_steps(rec: Recorder, run) -> None:
    """agent.react.step spans + server-reported mcp.rag.* stage spans, using the loop's measured start/duration."""
    for st in run.steps:
        rec.add("agent.react.step", st["start_ns"], st["ms"], tool=st["tool"], error=st.get("error", 0))
    hr = run.results.get("hybrid_retrieve")
    if hr:
        t = next(s["start_ns"] for s in run.steps if s["tool"] == "hybrid_retrieve")
        rec.add("mcp.rag.bm25", t, hr["bm25_ms"]); rec.add("mcp.rag.biencoder", t, hr["dense_ms"])
        rec.add("mcp.rag.rrf", t + int(max(hr["bm25_ms"], hr["dense_ms"]) * 1e6), hr["rrf_ms"])   # S1 ‖ S2, then fuse
    rr = next((s for s in run.steps if s["tool"] == "rerank" and "error" not in s), None)
    if rr:
        rec.add("mcp.rag.rerank", rr["start_ns"], rr["ms"])


def create_app(with_rag: bool = True, with_state: bool = True, budget: LoopBudget | None = None,
               llm="auto") -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.llm = make_client() if llm == "auto" else llm   # None -> extractive fallback
        rag = None
        if with_rag:
            rag = McpClient("hybrid-rag")
            await rag.start_stdio(str(ROOT / "mcp_servers" / "rag_server.py"))
            app.state.rag = rag
            graph = McpClient("graphify-vault")
            try:
                await graph.start_stdio(str(ROOT / "mcp_servers" / "graphify_server.py"))
                app.state.graph = graph
            except Exception as exc:  # noqa: BLE001  graph is an optimisation: degrade, never fail startup
                log.error("graphify unavailable: %s", exc)
        redis = None
        if with_state:
            try:
                st = McpClient("state-store")
                await st.start_http(STATE_URL)
                redis = aioredis.from_url(REDIS_URL)
                await redis.ping()
                app.state.state, app.state.redis = st, redis
                app.state.cache = AnswerCache(redis)
            except Exception as exc:  # noqa: BLE001
                log.error("state store unavailable, running degraded (Tier 3 fails closed): %s", exc)
                app.state.state = None
        if rag is not None:
            idem = RedisIdempotency(redis) if redis is not None else None
            app.state.loop = ReActLoop(ToolExecutionHarness(build_specs(rag, app.state.state, app.state.graph), app.state.plugins, idem),
                                       budget or budget_from_env())
        yield
        for c in (app.state.state, app.state.graph, app.state.rag):   # LIFO: anyio cancel scopes must exit in reverse order
            if c is not None:
                await c.stop()
        if app.state.redis is not None:
            await app.state.redis.aclose()
        if getattr(app.state.llm, "aclose", None):
            await app.state.llm.aclose()

    app = FastAPI(title="Reliable Assistant", version="0.4.0", lifespan=lifespan)
    app.state.plugins = build_plugins()
    app.state.chaos = []
    app.state.rag = app.state.state = app.state.redis = app.state.loop = app.state.llm = app.state.graph = app.state.cache = None

    @app.exception_handler(AssistantError)
    async def _assistant_error(_: Request, exc: AssistantError):
        return JSONResponse(exc.to_dict(), status_code=HTTP_STATUS.get(exc.code, 500))

    @app.get("/healthz")
    async def healthz(request: Request):
        return {"status": "ok", "stage": 5, "graph": request.app.state.graph is not None, "cache": request.app.state.cache is not None, "state_store": request.app.state.state is not None,
                "retrieval": request.app.state.rag is not None,
                "llm": request.app.state.llm is not None}

    @app.post("/api/v1/agent/query", response_model=AgentQueryOut)
    async def query(request: Request, authorization: str | None = Header(default=None),
                    x_tenant_id: str | None = Header(default=None)):
        try:
            body = AgentQueryIn.model_validate(await request.json())
        except (ValidationError, ValueError) as exc:
            raise AssistantError(-32602, str(exc)[:300]) from exc
        tenant, uhash = authenticate(authorization, body.tenant_id, x_tenant_id)
        trace_id = uuid.uuid4().hex
        ctx = {"tenant_id": tenant, "user_id_hash": uhash, "request_id": str(ULID()),
               "trace_id": trace_id, "message": body.message}
        app_ = request.app
        pm: PluginManager = app_.state.plugins
        rec = Recorder(trace_id)
        try:
            with rec.span("agent.query", tenant=tenant):
                return await _handle(app_, pm, rec, ctx, body, tenant, uhash, trace_id)
        finally:
            rec.flush(pm)

    async def _handle(app_, pm, rec, ctx, body, tenant, uhash, trace_id):
        try:
            with rec.span("hook.pre") as a:
                ctx.update(await asyncio.to_thread(pm.run_verdicts, "PRE_EXECUTION", ctx=ctx))   # pii -> injection (off-loop: may do IO)
                a["pii"] = sum(ctx.get("pii_counts", {}).values())
        except AssistantError as blocked:
            try:   # blocked requests are audited too — with independently redacted text, never raw
                safe = redact(body.message)[0]
            except Exception:  # noqa: BLE001
                safe = "[UNREDACTED-BLOCKED]"
            await audit(app_, {"meta": make_meta(ctx), "query_redacted": safe, "evidence_ids": [], "tool_calls": [],
                               "answer_redacted": "", "faithfulness": None,
                               "plugin_decisions": [{"blocked": blocked.name, "code": blocked.code}]})
            raise
        message = ctx["message"]
        decisions = [{"pii_counts": ctx.get("pii_counts", {}), "injection_score": ctx.get("injection_score", 0.0),
                  **({"injection_classifier": ctx["injection_classifier"]} if "injection_classifier" in ctx else {})}]

        session_id, version, prior = body.session_id or str(ULID()), 0, None
        if app_.state.state is not None and body.session_id:
            got = await app_.state.state.call("state_get", {"meta": make_meta(ctx), "session_id": session_id}, StateGetOut)
            prior, version = got.state, got.version

        def reply(answer, *, abstained, route, citations=(), ticket=None, halt=None, domain="general", faith=None):
            return AgentQueryOut(answer=DISCLAIMERS.get(domain, "") + answer, citations=list(citations), route=route,
                                 tenant_id=tenant, faithfulness=faith, abstained=abstained, trace_id=trace_id,
                                 hitl_ticket_id=ticket, halt_reason=halt, session_id=session_id)

        with rec.span("triage.classify") as a:
            tri = run_triage(message)
            a["route"] = tri.route
        run, evidence, facts, cache_hit = None, [], [], False
        anchor = GoalAnchorBlock(original_intent=message, normalized_entities=extract_entities(message),
                                 policy_constraints=POLICY_CONSTRAINTS)
        cache: AnswerCache | None = app_.state.cache
        cached = None
        if cache is not None and tri.route == "HYBRID_RAG":
            with rec.span("cache.lookup") as a:
                cached = await cache.get(tenant, uhash, message)
                a["hit"] = cached is not None

        if cached is not None:                         # ADR-3 hit: skip retrieval + generation (§1.3 step 3)
            cache_hit = True
            out = AgentQueryOut(**{**cached, "trace_id": trace_id, "session_id": session_id})
            evidence = [c["chunk_id"] for c in cached["citations"]]
        elif tri.route == "REJECT_OOS":
            out = reply(REFUSAL, abstained=True, route=tri.route, halt="REJECT_OOS")
        elif tri.route == "DIRECT_ANSWER":
            out = reply(GREETING, abstained=False, route=tri.route)
        elif app_.state.loop is None:
            out = reply(ABSTAIN, abstained=True, route=tri.route)
        else:
            run = await app_.state.loop.run(ctx, anchor, tri)
            _record_steps(rec, run)
            if run.state == State.HALTED_HITL:
                out = reply(f"Your request needs human approval. Ticket {run.hitl_ticket_id} is pending; "
                            "no action has been taken.", abstained=False, route=tri.route,
                            ticket=run.hitl_ticket_id, halt="PENDING_APPROVAL", domain=tri.domain)
            elif run.state == State.HALTED_ERROR and run.halt_code == -32002 and tri.route == "TRANSACTION_EXEC":
                raise AssistantError(-32002, "approval service unavailable; request not accepted (fail closed)")
            elif run.state == State.DONE and "rerank" in run.results:
                rr = RerankOut.model_validate(run.results["rerank"])
                if rr.abstain or not rr.ranked:                              # NG-5
                    out = reply(ABSTAIN, abstained=True, route=tri.route, halt="UNGROUNDED")
                else:
                    pool = run.results["hybrid_retrieve"]["chunks"]
                    by_id = {c["chunk_id"]: c for c in pool}
                    kept = [by_id[cid] for cid, _ in rr.ranked]
                    related = linked_as_chunks(run.results.get("graphify_expand", {}).get("nodes", []),
                                               {c["chunk_id"] for c in kept})
                    all_ev = kept + related
                    ev_by_id = {c["chunk_id"]: c for c in all_ev}
                    decisions.append({"tokens_prompt_evidence": sum(approx_tokens(c["text"]) for c in all_ev),
                                      "tokens_raw_pool": sum(approx_tokens(c["text"]) for c in pool),
                                      "graph_nodes": len(related)})
                    gen = None
                    if app_.state.llm is not None:
                        try:
                            with rec.span("llm.generate") as a:
                                gen = await generate_grounded(app_.state.llm, GEN_MODEL, VERIFY_MODEL, message,
                                                              POLICY_CONSTRAINTS, all_ev)
                                a["status"], a["llm_ms"] = gen["status"], round(gen["llm_ms"])
                        except AssistantError as exc:      # LLM down/rate-limited -> deterministic extractive fallback
                            log.warning("llm unavailable, extractive fallback: %s", exc)
                    if gen is not None and gen["status"] != "ok":
                        out = reply(ABSTAIN, abstained=True, route=tri.route, halt="UNGROUNDED", faith=gen["faithfulness"])
                        decisions.append({"llm": gen["status"], "faithfulness": gen["faithfulness"]})
                    else:
                        if gen is None:      # extractive answer: best chunk text, citing every kept chunk
                            text, cite_ids, faith = kept[0]["text"], [c["chunk_id"] for c in kept], None
                        else:
                            text, cite_ids, faith = gen["answer"], gen["citations"], gen["faithfulness"]
                            decisions.append({"llm": "ok", "faithfulness": faith, "llm_ms": round(gen["llm_ms"])})
                        evidence = cite_ids
                        ctx["evidence_texts"] = [ev_by_id[c]["text"] for c in cite_ids]
                        out = reply(text, abstained=False, route=tri.route, domain=tri.domain, faith=faith,
                                    citations=[Citation(chunk_id=cid, uri=ev_by_id[cid].get("source_uri", f"rag://chunks/{cid}"),
                                                        quote_span=(0, len(ev_by_id[cid]["text"]))) for cid in cite_ids])
                        facts = [SessionFact(fact=redact(text)[0][:300], source_chunk_id=cite_ids[0])]
            else:
                out = reply(ABSTAIN, abstained=True, route=tri.route,
                            halt=run.state.value if run.state != State.DONE else "NO_EVIDENCE")

        # POST_EXECUTION (never blocks the answer): lexical audit; flags needs_review; gates the cache write.
        with rec.span("hook.post.ragas") as a:
            pm.run_observers("POST_EXECUTION", ctx=ctx, answer=out.model_dump())
            post = ctx.get("post_audit")
            if post:
                a.update({k: v for k, v in post.items() if v is not None})
                decisions.append({"post_audit": post})
        if (cache is not None and not cache_hit and run is not None and not out.abstained and out.citations
                and tri.route == "HYBRID_RAG" and post and not post["needs_review"]):
            stored = await cache.put(tenant, uhash, message,
                                     {k: v for k, v in out.model_dump(mode="json").items()
                                      if k in ("answer", "citations", "route", "tenant_id", "faithfulness", "abstained")},
                                     out.faithfulness)
            decisions.append({"cache_write": stored})
        if cache_hit:
            decisions.append({"cache": "hit"})

        answer_red = redact(out.answer)[0]               # NG-6: nothing raw reaches persistence
        if run is not None:
            decisions.append({"route": tri.route, "state": run.state.value, "halt_code": run.halt_code,
                              "steps": run.iterations})
        await audit(app_, {"meta": make_meta(ctx), "query_redacted": message, "evidence_ids": evidence,
                           "tool_calls": (run.steps if run else []), "answer_redacted": answer_red,
                           "faithfulness": out.faithfulness, "plugin_decisions": decisions}, tri.domain)
        if app_.state.state is not None:
            turns = ((prior.last_turns if prior else []) + [{"q": message, "a": answer_red[:300]}])[-3:]
            new_state = SessionState(session_id=session_id, turn_index=min(500, (prior.turn_index + 1) if prior else 0),
                                     goal_anchor=anchor, last_turns=turns)
            await app_.state.state.call("state_put", {"meta": make_meta(ctx), "state": new_state.model_dump(mode="json"),
                                                      "expected_version": version,
                                                      "new_facts": [f.model_dump(mode="json") for f in facts]}, StatePutOut)
        return out

    # ---- HITL reviewer API (Streamlit Tab 1 queue consumes this) ----
    @app.get("/api/v1/hitl/pending")
    async def hitl_pending(request: Request, authorization: str | None = Header(default=None)):
        tenant, uhash, role = authenticate(authorization, with_role=True)
        if role != "hitl_reviewer":
            raise AssistantError(-32012, "reviewer role required")
        ctx = {"tenant_id": tenant, "user_id_hash": uhash, "request_id": str(ULID()), "trace_id": uuid.uuid4().hex}
        res = await _state(request).call_raw("hitl_pending", {"meta": make_meta(ctx)})
        return {"tickets": res.get("result", res)}

    @app.post("/api/v1/hitl/{ticket_id}/decision")
    async def hitl_decision(ticket_id: str, d: Decision, request: Request, authorization: str | None = Header(default=None)):
        tenant, uhash, role = authenticate(authorization, with_role=True)
        if role != "hitl_reviewer" or d.decision not in ("approve", "reject"):
            raise AssistantError(-32012 if role != "hitl_reviewer" else -32602, "reviewer role / valid decision required")
        ctx = {"tenant_id": tenant, "user_id_hash": uhash, "request_id": str(ULID()), "trace_id": uuid.uuid4().hex}
        res = await _state(request).call_raw("hitl_decide", {"params": {"meta": make_meta(ctx), "ticket_id": ticket_id,
                                                                        "approve": d.decision == "approve"}})
        await audit(request.app, {"meta": make_meta(ctx), "query_redacted": f"HITL {d.decision} {ticket_id}",
                                  "evidence_ids": [], "tool_calls": [], "answer_redacted": res.get("status", ""),
                                  "faithfulness": None, "plugin_decisions": [{"hitl_decision": d.decision}]})
        return {"ticket_id": ticket_id, **res, "executed": False}   # approval only records the decision (see notes)

    # ---- read-only audit view for the compliance auditor role (tenant-scoped; redacted records only) ----
    @app.get("/api/v1/audit/recent")
    async def audit_recent(request: Request, limit: int = 50, authorization: str | None = Header(default=None)):
        tenant, uhash, role = authenticate(authorization, with_role=True)
        if role != "auditor":
            raise AssistantError(-32012, "auditor role required")
        ctx = {"tenant_id": tenant, "user_id_hash": uhash, "request_id": str(ULID()), "trace_id": uuid.uuid4().hex}
        st = _state(request)
        recs = await st.call("audit_recent", {"meta": make_meta(ctx), "limit": max(1, min(limit, 200))}, AuditRecentOut)
        return {"records": recs.records, "chain": await st.call_raw("audit_verify", {"meta": make_meta(ctx)})}

    def _state(request: Request) -> McpClient:
        if request.app.state.state is None:
            raise AssistantError(-32002, "state store unavailable")
        return request.app.state.state

    if os.environ.get("RA_DEV") == "1":      # developer control plane: opt-in, role=developer only
        from .devapi import build_router
        app.include_router(build_router())
    return app


app = create_app()
