"""Generates docs/workflow_orchestration.png (Gate 2) from design.md section 1.3."""
import os
from graphviz import Digraph

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_orchestration")
g = Digraph("workflow", format="png")
g.attr(rankdir="TB", dpi="300", fontname="Helvetica", fontsize="20", labelloc="t", pad="0.3",
       nodesep="0.3", ranksep="0.55", compound="true", splines="polyline",
       label="The Unreliable Assistant -> Reliable Assistant v1: Workflow Orchestration (design.md 1.3)")
g.attr("node", shape="box", style="rounded,filled", fontname="Helvetica", fontsize="11", fillcolor="white")
g.attr("edge", fontname="Helvetica", fontsize="9")

def cluster(name, label, color, nodes):
    with g.subgraph(name="cluster_" + name) as c:
        c.attr(label=label, style="rounded,filled", color="#888888", fillcolor=color, fontsize="13")
        for nid, nl, *extra in nodes:
            c.node(nid, nl, **(extra[0] if extra else {}))

g.node("client", "Clients\n50K VUs (k6 + JMeter)\nHTTPS/2 + SSE", shape="oval", fillcolor="#dddddd")

cluster("ingress", "1. Ingress & Security Layer", "#fdecea", [
    ("gw", "NGINX/Envoy + FastAPI Gateway\nJWT -> tenant, rate-limit, admission"),
    ("pii", "PII Redactor (p10)\nfail-closed"),
    ("inj", "Injection Classifier (p20)\nfail-closed"),
    ("cache", "Redis Semantic Cache Check\nSHA256(tenant:user:norm_query)\nTTL 300s +/- 45s", {"shape": "diamond", "fillcolor": "#ffffff"}),
])

cluster("triage", "2. Triage & Subagent Routing", "#e8f1fb", [
    ("tri", "Triage Agent\ntemperature=0.0, <= 25 ms\n(intent, domain, risk_tier, complexity)"),
    ("direct", "DIRECT_ANSWER\nDirect Response"),
    ("oos", "REJECT_OOS\nDeterministic refusal"),
    ("ret", "Retrieval Subagent\nread-only"),
    ("exe", "Execution Subagent\nTier 3 -> HITL freeze"),
    ("ver", "Verification Subagent\nNLI claim-to-chunk, veto"),
])

cluster("mcp", "3. MCP (JSON-RPC 2.0) & 3-Stage RAG Pipeline", "#eaf6ea", [
    ("bm25", "Stage 1: BM25\nk1=1.25 b=0.75 top50 (<12 ms)"),
    ("dense", "Stage 2: Bi-Encoder + FAISS HNSW\nbge-large ONNX top50 (<25 ms)"),
    ("rrf", "RRF k=60\ndedup chunk_hash, pool 40 (<3 ms)"),
    ("ce", "Stage 3: Cross-Encoder\nbge-reranker-v2-m3 FP16, top4\nscore > 0.42 (<45 ms)"),
    ("abst", "Abstention short-circuit\ntop score < 0.42, <90 ms\n(no LLM call)", {"fillcolor": "#fff4cc"}),
    ("gfy", "mcp-graphify-vault\nObsidian + Graphify, depth<=2\n~320 tok"),
    ("st", "mcp-state-store\nsession_facts / audit / workflow"),
])

cluster("ctrl", "4. Control Plane & Egress", "#f3ebfa", [
    ("loop", "Bounded ReAct Loop (max_steps=5)\nstep 120 ms, loop 450 ms\nGoalAnchorBlock | obs <=180 tok | window=3"),
    ("harness", "ToolExecutionHarness\nPydantic -> auto-repair -> RBAC tier -> UUIDv4"),
    ("cb", "Circuit Breaker\n>=2 identical failures -> HALTED_BREAKER"),
    ("hitl", "Tier 3 HITL Gate\nPENDING_APPROVAL", {"fillcolor": "#ffe0b2"}),
    ("llm", "LLM generate\nvLLM/TensorRT-LLM"),
    ("post", "Post-Execution\nRAGAS audit (faithfulness >= 0.92)\nhash-chained audit log"),
    ("otel", "OTel Export (W3C traceparent)\n-> Collector -> Tempo/Prom/Grafana"),
    ("resp", "Response\n+ citations + faithfulness", {"shape": "oval", "fillcolor": "#dddddd"}),
])

g.node("pg", "PostgreSQL + pgvector\nRLS by tenant_id", shape="cylinder", fillcolor="#fffbe6")
g.node("redis", "Redis Cluster\ntenant hash-tagged keys", shape="cylinder", fillcolor="#fffbe6")

e = g.edge
e("client", "gw"); e("gw", "pii"); e("pii", "inj"); e("inj", "cache")
e("cache", "resp", label="hit (p95<=120 ms)", style="dashed")
e("cache", "tri", label="miss")
e("tri", "direct", label="DIRECT_ANSWER"); e("tri", "ret", label="HYBRID_RAG")
e("tri", "exe", label="TRANSACTION_EXEC"); e("tri", "oos", label="REJECT_OOS")
e("tri", "ver", style="invis")
e("oos", "resp", style="dashed"); e("direct", "llm")
e("ret", "loop"); e("exe", "loop")
e("loop", "harness", label="tool call"); e("harness", "cb")
e("cb", "bm25", label="mcp-hybrid-rag\n(150 ms)", lhead="cluster_mcp")
e("bm25", "rrf"); e("dense", "rrf"); e("rrf", "ce")
e("ce", "abst", label="< 0.42"); e("abst", "resp", style="dashed")
e("ce", "gfy", style="dotted", label="seeds")
e("ce", "llm", label=">= 0.42, top4 + graph")
e("cb", "st", style="dotted")
e("cb", "hitl", label="Tier 3"); e("hitl", "st", label="PENDING_APPROVAL")
e("llm", "ver"); e("ver", "post", label="faithfulness"); e("post", "resp")
e("post", "otel")
e("ver", "llm", label="veto -> abstain", style="dashed", constraint="false")
e("gw", "redis", style="dotted", dir="both"); e("cache", "redis", style="dotted", dir="both")
e("loop", "redis", style="dotted", dir="both"); e("st", "pg", dir="both"); e("post", "pg", style="dotted")

g.render(out, cleanup=True)
print("wrote", out + ".png")
