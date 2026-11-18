import asyncio
import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import reliable_assistant.api as api
from reliable_assistant.agent.harness import ToolExecutionHarness, ToolSpec
from reliable_assistant.agent.loop import DEV_BUDGET, ReActLoop, State
from reliable_assistant.agent.triage import triage
from reliable_assistant.auth import mint_token
from reliable_assistant.cache import cache_key, norm_query, ttl_s
from reliable_assistant.llm.ground import build_messages, generate_grounded, linked_as_chunks
from reliable_assistant.quality import audit_answer
from reliable_assistant.retrieval.graph import VaultGraph
from reliable_assistant.telemetry import spans_for
from reliable_assistant.types import (GoalAnchorBlock, GraphifyExpandIn, HybridRetrieveIn, RerankIn)

VAULT = Path(__file__).resolve().parents[1] / "knowledge_vault"


# ---------------- graph ----------------
def test_graph_expands_wikilinks_within_tenant_and_budget():
    g = VaultGraph("acme", VAULT)
    r = g.expand(["acme:lot-recall-sop:0"], 2, 320, ["wikilink"])
    assert [n["id"] for n in r["nodes"]] == ["sku-labeling-rules"] and r["nodes"][0]["hop"] == 1
    assert r["tokens_used"] <= 320
    assert g.expand(["acme:lot-recall-sop:0"], 2, 5, ["wikilink"])["nodes"] == []        # budget respected
    back = g.expand(["acme:sku-labeling-rules:0"], 1, 320, ["wikilink", "backlink"])
    assert [n["id"] for n in back["nodes"]] == ["lot-recall-sop"]
    assert VaultGraph("globex", VAULT).expand(["acme:lot-recall-sop:0"], 2, 320, ["wikilink"])["nodes"] == []


def test_graph_rejects_path_traversal(tmp_path):
    (tmp_path / "t1").mkdir(); (tmp_path / "secret.md").write_text("# secret")
    assert VaultGraph("../", tmp_path).notes == {} and VaultGraph("t1", tmp_path).notes == {}
    with pytest.raises(Exception):
        GraphifyExpandIn(meta=_meta(), seeds=["../etc/passwd"])


def _meta(t="acme"):
    return dict(tenant_id=t, user_id_hash="a" * 32, request_id="0" * 26, trace_id="b" * 32, deadline_ms=500,
                idempotency_key=str(uuid.uuid4()))


# ---------------- loop: parallel step + graceful degradation ----------------
CHUNK = {"chunk_id": "acme:lot-recall-sop:0", "chunk_hash": "a" * 64, "text": "t", "bm25_score": 1.0, "cosine_score": .5,
         "rrf_score": .03, "source_uri": "rag://chunks/acme:lot-recall-sop:0", "acl_fingerprint": "a" * 16}


def fake_harness(graph_ok: bool):
    order = []

    async def hyb(m): order.append("hyb"); return {"chunks": [CHUNK], "bm25_ms": 1, "dense_ms": 1, "rrf_ms": 1, "degraded": "none"}
    async def rer(m): order.append("rer"); await asyncio.sleep(0.05); return {"ranked": [[CHUNK["chunk_id"], .9]], "top_score": .9, "abstain": False, "precision": "fp32", "degraded": False}
    async def gra(m):
        order.append("gra")
        if not graph_ok:
            raise RuntimeError("graph down")
        return {"nodes": [], "tokens_used": 0, "tokens_saved_vs_raw": 0}
    specs = [ToolSpec("hybrid_retrieve", "TIER_1_READ", HybridRetrieveIn, hyb), ToolSpec("rerank", "TIER_1_READ", RerankIn, rer),
             ToolSpec("graphify_expand", "TIER_1_READ", GraphifyExpandIn, gra)]
    return ToolExecutionHarness(specs, api.build_plugins()), order


CTX = {"tenant_id": "acme", "user_id_hash": "a" * 32, "request_id": "0" * 26, "trace_id": "b" * 32}
ANCHOR = GoalAnchorBlock(original_intent="How fast must LOT-88392 be quarantined?", normalized_entities={}, policy_constraints=[])


@pytest.mark.parametrize("graph_ok", [True, False])
async def test_rerank_and_graph_run_in_one_step_and_graph_is_optional(graph_ok):
    h, order = fake_harness(graph_ok)
    run = await ReActLoop(h, DEV_BUDGET).run(dict(CTX), ANCHOR, triage(ANCHOR.original_intent))
    assert run.state == State.DONE and "rerank" in run.results
    assert run.iterations == 2                                # retrieve; then rerank ‖ graph  (KPI-17 ≤ 2.2)
    assert ("graphify_expand" in run.results) == graph_ok
    assert order[0] == "hyb" and set(order[1:]) == {"rer", "gra"}


# ---------------- quality audit ----------------
def test_audit_flags_ungrounded_and_scores_entities():
    ev = ["When LOT-88392 is on hold, quarantine all remaining units within 4 hours and tag the bin red."]
    good = audit_answer("How fast for LOT-88392?", "Quarantine all remaining units within 4 hours.", ev)
    bad = audit_answer("How fast for LOT-88392?", "Ship the units immediately to retail customers overseas.", ev)
    assert good["groundedness"] >= 0.8 and good["entity_recall"] == 1.0
    assert bad["groundedness"] < 0.5
    assert audit_answer("hi", "x", [])["entity_recall"] is None


# ---------------- cache keys ----------------
def test_cache_key_isolated_per_user_and_normalised():
    assert cache_key("acme", "u1", "How  FAST is lot 44?") == cache_key("acme", "u1", "how fast is LOT-00044?")
    assert cache_key("acme", "u1", "q") != cache_key("acme", "u2", "q") != cache_key("globex", "u1", "q")
    assert cache_key("acme", "u1", "q").startswith("{acme}:cache:")
    assert all(255 <= ttl_s() <= 345 for _ in range(200))


async def test_linked_notes_are_citable_evidence():
    class L:
        async def chat(self, model, messages, max_tokens=400, json_mode=False):
            self.msgs = messages
            return ('{"supported": [true]}' if json_mode else "Labels go on the top-right face [acme:sku-labeling-rules:0]."), 1.0
    nodes = [{"chunk_id": "acme:sku-labeling-rules:0", "summary": "Labels go on the top-right face of the carton."}]
    ev = [{"chunk_id": "acme:lot-recall-sop:0", "text": "Quarantine in 4 hours."}] + linked_as_chunks(nodes, {"acme:lot-recall-sop:0"})
    llm = L()
    r = await generate_grounded(llm, "g", "v", "q", ["c"], ev)
    assert r["status"] == "ok" and r["citations"] == ["acme:sku-labeling-rules:0"]
    assert 'source="linked"' in build_messages("q", ["c"], ev)[1]["content"]


# ---------------- end to end: cache + spans + audit (real Postgres/Redis, fake LLM) ----------------
class FakeLlm:
    def __init__(self): self.gen_calls = 0
    async def chat(self, model, messages, max_tokens=400, json_mode=False):
        if json_mode:
            return '{"supported": [true]}', 1.0
        self.gen_calls += 1
        return "Units must be quarantined within 4 hours [acme:lot-recall-sop:0].", 1.0


@pytest.fixture
def e2e(infra):
    llm = FakeLlm()
    with TestClient(api.create_app(with_rag=True, with_state=True, llm=llm)) as c:
        h = c.get("/healthz").json()
        assert h["state_store"] and h["graph"] and h["cache"]
        yield c, llm


def hdr(t="acme", u="u1"): return {"Authorization": f"Bearer {mint_token(t, u)}"}


async def _last_audit(tenant):
    import asyncpg
    c = await asyncpg.connect("postgresql://ra:ra_dev_only@localhost:5433/ra")
    r = await c.fetchval("SELECT payload FROM audit_log WHERE tenant_id=$1 ORDER BY seq DESC LIMIT 1", tenant)
    await c.close()
    return json.loads(r) if isinstance(r, str) else r


def test_cache_miss_hit_isolation_spans_and_audit(e2e):
    c, llm = e2e
    q = {"message": f"How fast must LOT-88392 be quarantined? (case {uuid.uuid4().hex[:6]})"}
    r1 = c.post("/api/v1/agent/query", json=q, headers=hdr()).json()
    assert not r1["abstained"] and r1["faithfulness"] == 1.0 and llm.gen_calls == 1
    a1 = asyncio.run(_last_audit("acme"))
    dec = {k: v for d in a1["plugin_decisions"] for k, v in d.items()}
    assert dec["steps"] == 2 and dec["cache_write"] is True and dec["post_audit"]["needs_review"] is False
    assert dec["tokens_prompt_evidence"] <= dec["tokens_raw_pool"]
    # same user, same question -> served from cache, no new LLM call
    r2 = c.post("/api/v1/agent/query", json=q, headers=hdr()).json()
    assert llm.gen_calls == 1 and r2["answer"] == r1["answer"] and r2["trace_id"] != r1["trace_id"]
    assert any("cache" in d and d["cache"] == "hit" for d in asyncio.run(_last_audit("acme"))["plugin_decisions"])
    # another user in the same tenant must NOT hit user1's cache entry (ADR-3)
    c.post("/api/v1/agent/query", json=q, headers=hdr(u="someone-else"))
    assert llm.gen_calls == 2
    # spans for the first request
    names = {s.name for s in spans_for(r1["trace_id"])}
    assert {"agent.query", "hook.pre", "triage.classify", "cache.lookup", "agent.react.step", "mcp.rag.bm25",
            "mcp.rag.biencoder", "mcp.rag.rrf", "mcp.rag.rerank", "llm.generate", "hook.post.ragas"} <= names
    roots = [s for s in spans_for(r1["trace_id"]) if s.name == "agent.query"]
    assert len(roots) == 1
    assert all(s.parent is not None for s in spans_for(r1["trace_id"]) if s.name != "agent.query")


def test_abstention_and_unverified_answers_are_never_cached(e2e):
    c, llm = e2e
    j = c.post("/api/v1/agent/query", json={"message": f"What is the capital of France? {uuid.uuid4().hex[:6]}"}, headers=hdr()).json()
    assert j["abstained"]
    dec = {k: v for d in asyncio.run(_last_audit("acme"))["plugin_decisions"] for k, v in d.items()}
    assert "cache_write" not in dec
