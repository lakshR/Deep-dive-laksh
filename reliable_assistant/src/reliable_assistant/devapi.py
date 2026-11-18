"""Developer control-plane API for the Streamlit workbench. Mounted ONLY when RA_DEV=1, and every route requires a JWT
with role=developer. Reads are tenant-scoped to the caller's tenant; /metrics returns aggregate numbers only."""
import json
import os
import statistics
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel
from ulid import ULID

from .agent.loop import DEV_BUDGET, EXECUTION, Action, Final, LoopBudget, ReActLoop
from .auth import authenticate
from .errors import AssistantError
from .retrieval.graph import VaultGraph
from .telemetry import RING
from .types import (AuditRecentOut, GoalAnchorBlock, GraphifyExpandOut, HybridRetrieveOut, RerankOut, TriageOut)

VAULT = Path(os.environ.get("RA_VAULT", Path(__file__).resolve().parents[2] / "knowledge_vault"))


def _dev(authorization) -> tuple[str, str]:
    tenant, uhash, role = authenticate(authorization, with_role=True)
    if role != "developer":
        raise AssistantError(-32012, "developer role required")
    return tenant, uhash


def _ctx(tenant, uhash):
    return {"tenant_id": tenant, "user_id_hash": uhash, "request_id": str(ULID()), "trace_id": uuid.uuid4().hex}


def _pct(vals: list[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


class Toggle(BaseModel):
    enabled: bool


class RagLab(BaseModel):
    query: str
    sparse_top_k: int = 50
    dense_top_k: int = 50
    bm25_k1: float = 1.25
    bm25_b: float = 0.75
    rrf_k: int = 60
    pool: int = 40
    threshold: float = 0.42


class GraphReq(BaseModel):
    seeds: list[str]
    hops: int = 2
    token_budget: int = 320
    edge_types: list[str] = ["wikilink"]


class Tool(BaseModel):
    tool: str


class ChaosReq(BaseModel):
    mode: str = "alternating_sku"


class Alternating:
    """Chaos policy: retries the 'same' SKU lookup forever, alternating the raw spelling (SKU99 / SKU-099)."""
    def __init__(self): self.n = 0
    def next(self, run, anchor, triage):
        self.n += 1
        return Action("validate_sku", {"sku": "SKU99" if self.n % 2 else "SKU-099"}, EXECUTION)


def build_router() -> APIRouter:
    r = APIRouter(prefix="/api/v1/dev")
    from . import api as A   # late import (api includes this router)

    @r.get("/status")
    async def status(request: Request, authorization: str | None = Header(default=None)):
        _dev(authorization)
        st, out = request.app.state, []
        for name, c in (("hybrid-rag", st.rag), ("state-store", st.state), ("graphify-vault", st.graph)):
            tools, ok = [], False
            if c is not None:
                try:
                    tools, ok = [t.name for t in (await c.session.list_tools()).tools], True
                except Exception:  # noqa: BLE001
                    ok = False
            out.append({"server": name, "connected": ok, "tools": tools})
        pm = st.plugins
        plugins = [{"name": n, "stage": c["stage"], "priority": c.get("priority"), "fail_closed": c.get("fail_closed", False),
                    "enabled": pm.enabled[n], "registered": pm.pm.get_plugin(n) is not None} for n, c in pm.config.items()]
        return {"mcp": out, "plugins": plugins, "llm": st.llm is not None, "cache": st.cache is not None}

    @r.post("/plugins/{name}")
    async def toggle(name: str, t: Toggle, request: Request, authorization: str | None = Header(default=None)):
        tenant, uhash = _dev(authorization)
        pm = request.app.state.plugins
        if name not in pm.config:
            raise AssistantError(-32601, f"unknown plugin {name}")
        pm.set_enabled(name, t.enabled)
        ctx = _ctx(tenant, uhash)
        await A.audit(request.app, {"meta": A.make_meta(ctx), "query_redacted": f"DEV plugin {name} enabled={t.enabled}",
                                    "evidence_ids": [], "tool_calls": [], "answer_redacted": "", "faithfulness": None,
                                    "plugin_decisions": [{"dev_toggle": name, "enabled": t.enabled}]})
        return {"name": name, "enabled": t.enabled}

    @r.get("/jsonrpc")
    async def jsonrpc(request: Request, authorization: str | None = Header(default=None)):
        _dev(authorization)
        st = request.app.state
        rows = [e for c in (st.rag, st.state, st.graph) if c is not None for e in c.log]
        return {"calls": sorted(rows, key=lambda e: e["ts"], reverse=True)[:60]}

    @r.get("/traces")
    async def traces(authorization: str | None = Header(default=None)):
        tenant, _ = _dev(authorization)
        roots = [s for s in RING.spans if s.name == "agent.query" and s.attributes.get("tenant") == tenant]
        return {"traces": [{"trace_id": format(s.context.trace_id, "032x"), "ms": round((s.end_time - s.start_time) / 1e6, 1),
                            "start_ns": s.start_time} for s in sorted(roots, key=lambda s: -s.start_time)[:30]]}

    @r.get("/traces/{trace_id}")
    async def trace(trace_id: str, authorization: str | None = Header(default=None)):
        tenant, _ = _dev(authorization)
        spans = [s for s in RING.spans if format(s.context.trace_id, "032x") == trace_id]
        if not any(s.name == "agent.query" and s.attributes.get("tenant") == tenant for s in spans):
            raise AssistantError(-32012, "trace not found for this tenant")
        return {"spans": [{"name": s.name, "span_id": format(s.context.span_id, "016x"),
                           "parent_id": format(s.parent.span_id, "016x") if s.parent else None, "start_ns": s.start_time,
                           "ms": round((s.end_time - s.start_time) / 1e6, 2), "attrs": dict(s.attributes)}
                          for s in sorted(spans, key=lambda s: s.start_time)]}

    @r.post("/rag-lab")
    async def rag_lab(p: RagLab, request: Request, authorization: str | None = Header(default=None)):
        tenant, uhash = _dev(authorization)
        rag, ctx = request.app.state.rag, _ctx(tenant, uhash)
        meta = lambda: {**A.make_meta(ctx)}
        ret = await rag.call("hybrid_retrieve", {"meta": meta(), "query": p.query, "sparse_top_k": p.sparse_top_k,
                                                 "dense_top_k": p.dense_top_k, "bm25_k1": p.bm25_k1, "bm25_b": p.bm25_b,
                                                 "rrf_k": p.rrf_k, "pool": p.pool}, HybridRetrieveOut)
        if not ret.chunks:
            return {"rows": [], "timing": {}, "abstain": True, "top_score": 0.0}
        rr = await rag.call("rerank", {"meta": meta(), "query": p.query, "threshold": p.threshold,
                                       "candidates": [c.model_dump(mode="json") for c in ret.chunks]}, RerankOut)
        ce = dict(rr.all_scores)
        rank = lambda key: {c.chunk_id: i for i, c in enumerate(sorted([c for c in ret.chunks if getattr(c, key) is not None],
                                                                         key=lambda c: -getattr(c, key)), 1)}
        bm, cos, fused = rank("bm25_score"), rank("cosine_score"), rank("rrf_score")
        ce_rank = {cid: i for i, (cid, _) in enumerate(rr.all_scores, 1)}
        rows = [{"chunk_id": c.chunk_id, "bm25": c.bm25_score, "bm25_rank": bm.get(c.chunk_id), "cosine": c.cosine_score,
                 "dense_rank": cos.get(c.chunk_id), "rrf": c.rrf_score, "rrf_rank": fused[c.chunk_id],
                 "cross_encoder": ce.get(c.chunk_id), "ce_rank": ce_rank.get(c.chunk_id),
                 "kept": c.chunk_id in {k for k, _ in rr.ranked}, "text": c.text[:160]} for c in ret.chunks]
        return {"rows": sorted(rows, key=lambda x: x["ce_rank"] or 999), "abstain": rr.abstain, "top_score": rr.top_score,
                "timing": {"bm25_ms": ret.bm25_ms, "dense_ms": ret.dense_ms, "rrf_ms": ret.rrf_ms}}

    @r.post("/graph")
    async def graph(p: GraphReq, request: Request, authorization: str | None = Header(default=None)):
        tenant, uhash = _dev(authorization)
        if request.app.state.graph is None:
            raise AssistantError(-32002, "graphify unavailable")
        out = await request.app.state.graph.call("graphify_expand", {"meta": A.make_meta(_ctx(tenant, uhash)), "seeds": p.seeds,
                                                                     "hops": p.hops, "token_budget": p.token_budget,
                                                                     "edge_types": p.edge_types}, GraphifyExpandOut)
        return out.model_dump(mode="json")

    @r.get("/notes")
    async def notes(authorization: str | None = Header(default=None)):
        tenant, _ = _dev(authorization)
        g = VaultGraph(tenant, VAULT)
        return {"notes": [{"slug": n.slug, "meta": n.meta, "links": sorted(n.links), "body": n.body} for n in g.notes.values()]}

    @r.get("/sessions")
    async def sessions(request: Request, authorization: str | None = Header(default=None)):
        tenant, _ = _dev(authorization)
        redis, out = request.app.state.redis, []
        if redis is None:
            return {"sessions": []}
        async for key in redis.scan_iter(match=f"\\{{{tenant}\\}}:sess:*", count=200):
            h = await redis.hgetall(key)
            out.append({"key": key.decode().split(":sess:")[1], "version": int(h.get(b"ver", 0)),
                        "ttl_s": await redis.ttl(key), "state": json.loads(h[b"state"]) if b"state" in h else None})
            if len(out) >= 1000:
                break
        out.sort(key=lambda x: -x["ttl_s"])        # TTL resets on every write: highest TTL == most recently active
        return {"sessions": out[:50]}

    async def _audit(request, tenant, uhash, limit):
        res = await request.app.state.state.call("audit_recent", {"meta": A.make_meta(_ctx(tenant, uhash)), "limit": limit}, AuditRecentOut)
        return res.records

    @r.get("/audit")
    async def audit(request: Request, limit: int = 50, authorization: str | None = Header(default=None)):
        tenant, uhash = _dev(authorization)
        if request.app.state.state is None:
            raise AssistantError(-32002, "state store unavailable")
        recs = await _audit(request, tenant, uhash, min(limit, 500))
        v = await request.app.state.state.call_raw("audit_verify", {"meta": A.make_meta(_ctx(tenant, uhash))})
        return {"records": recs, "chain": v}

    @r.get("/breaker")
    async def breaker_state(request: Request, authorization: str | None = Header(default=None)):
        _dev(authorization)
        b = request.app.state.plugins.pm.get_plugin("circuit_breaker")
        return {"forced_open": sorted(b.forced), "events": list(b.events)[-30:], "max_identical_calls": b.max_identical}

    @r.post("/breaker/trip")
    async def breaker_trip(t: Tool, request: Request, authorization: str | None = Header(default=None)):
        _dev(authorization)
        request.app.state.plugins.pm.get_plugin("circuit_breaker").trip(t.tool)
        return {"tripped": t.tool}

    @r.post("/breaker/reset")
    async def breaker_reset(request: Request, t: Tool | None = None, authorization: str | None = Header(default=None)):
        _dev(authorization)
        request.app.state.plugins.pm.get_plugin("circuit_breaker").reset(t.tool if t else None)
        return {"reset": t.tool if t else "all"}

    @r.post("/chaos/loop")
    async def chaos(c: ChaosReq, request: Request, authorization: str | None = Header(default=None)):
        tenant, uhash = _dev(authorization)
        loop = request.app.state.loop
        if loop is None:
            raise AssistantError(-32002, "agent loop unavailable")
        tri = TriageOut(intent="chaos", domain="general", risk_tier="TIER_2_COMPUTE", complexity="low", route="HYBRID_RAG")
        anchor = GoalAnchorBlock(original_intent="chaos retry loop", normalized_entities={}, policy_constraints=[])
        run = await ReActLoop(loop.harness, LoopBudget(max_iterations=50, step_timeout_ms=5000, total_ms=30000), Alternating()).run(
            _ctx(tenant, uhash), anchor, tri)
        executed = sum(1 for s in run.steps if "error" not in s)
        res = {"state": run.state.value, "halt_code": run.halt_code, "executed": executed, "attempts": len(run.steps),
               "steps": [{"tool": s["tool"], "ms": round(s["ms"], 1), "error": s.get("error")} for s in run.steps]}
        request.app.state.chaos.append(res["state"] == "HALTED_BREAKER")
        return res

    @r.get("/metrics")
    async def metrics(request: Request, authorization: str | None = Header(default=None)):
        tenant, uhash = _dev(authorization)
        st, m = request.app.state, {}
        by: dict[str, list[float]] = {}
        for s in RING.spans:
            by.setdefault(s.name, []).append((s.end_time - s.start_time) / 1e6)
        for name, v in by.items():
            for q, lab in ((.5, "p50"), (.95, "p95"), (.99, "p99")):
                m[f"span.{name}.{lab}"] = {"value": round(_pct(v, q), 2), "n": len(v), "note": "dev hardware, all tenants"}
        h = st.loop.harness.metrics if st.loop else {}
        ok, bad = h.get("first_pass_ok", 0), h.get("first_pass_fail", 0)
        if ok + bad:
            m["harness.first_pass_pct"] = {"value": round(100 * ok / (ok + bad), 1), "n": ok + bad}
        rok, rbad = h.get("repair_ok", 0), h.get("repair_fail", 0)
        if rok + rbad:
            m["harness.repair_pct"] = {"value": round(100 * rok / (rok + rbad), 1), "n": rok + rbad}
        chaos = list(st.chaos)
        if chaos:
            m["chaos.escape_pct"] = {"value": round(100 * sum(chaos) / len(chaos), 1), "n": len(chaos)}
        if st.state is not None:
            recs = [x["payload"] for x in await _audit(request, tenant, uhash, 300)]   # recent window
            dec = [{k: v for d in x.get("plugin_decisions", []) for k, v in d.items()} for x in recs]
            steps = [d["steps"] for d in dec if d.get("state") == "DONE" and "steps" in d]
            if steps:
                m["audit.steps_mean"] = {"value": round(statistics.mean(steps), 2), "n": len(steps)}
            f = [x["faithfulness"] for x in recs if x.get("faithfulness") is not None]
            if f:
                m["audit.faith_mean"] = {"value": round(statistics.mean(f), 3), "n": len(f)}
            tx = [d for d in dec if d.get("route") == "TRANSACTION_EXEC"]
            if tx:
                m["audit.hitl_pct"] = {"value": round(100 * sum(d.get("state") == "HALTED_HITL" for d in tx) / len(tx), 1), "n": len(tx)}
            ev = sum(d.get("tokens_prompt_evidence", 0) for d in dec); raw = sum(d.get("tokens_raw_pool", 0) for d in dec)
            n_tok = sum(1 for d in dec if "tokens_raw_pool" in d)
            if raw:
                m["audit.token_reduction_pct"] = {"value": round(100 * (1 - ev / raw), 1), "n": n_tok}
            hits = sum(1 for d in dec if d.get("cache") == "hit"); looked = hits + sum(1 for d in dec if "cache_write" in d)
            if looked:
                m["audit.cache_hit_pct"] = {"value": round(100 * hits / looked, 1), "n": looked, "note": "not steady state"}
        ev = Path(__file__).resolve().parents[2] / "evals" / "results" / "latest.json"
        if ev.exists():                      # offline eval results (author-generated corpora) — see evals/run_evals.py
            for k, v in json.loads(ev.read_text())["metrics"].items():
                m[k] = {**v, "note": v.get("note", "")[:90]}
            m["eval.recall50"] = m.get("eval.recall50_pct", {}) and {**m["eval.recall50_pct"]}
        return {"metrics": m}

    @r.post("/chaos/reset-metrics")
    async def reset_metrics(request: Request, authorization: str | None = Header(default=None)):
        _dev(authorization)
        request.app.state.chaos.clear(); RING.spans.clear()
        return {"ok": True}

    return r
