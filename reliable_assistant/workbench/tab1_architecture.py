import streamlit as st

from reliable_assistant.agent.triage import triage

from .common import ROOT, api, need_dev

# Maintained BY HAND — honest status of each blueprint gate in this repo.
GATES = [
    ("Gate 0 — stack, MCP servers, plugins", "🟡 Partial", "All 3 MCP servers + 6 plugins. Python 3.14 (doc: 3.11); small CPU models instead of bge-large/reranker-v2-m3; FP32, no ONNX; Redis/Postgres single-node."),
    ("Gate 1 — design.md", "🟡 Partial", "Written; not tagged `design-locked`; Appendix B open items unresolved."),
    ("Gate 2 — workflow_orchestration.png", "✅ Done", "docs/workflow_orchestration.png"),
    ("Gates 3 & 8 — 3-stage RAG, rerank, RAGAS", "🟡 Partial", "BM25 + HNSW + RRF + cross-encoder + 0.42 abstention work. RAGAS replaced by a lexical proxy; threshold not calibrated for MiniLM."),
    ("Gate 4 — harness & loop", "✅ Done", "Validation, auto-repair, RBAC, idempotency, bounded loop, circuit breaker (canonical hash)."),
    ("Gate 5 — context retention", "🟡 Partial", "GoalAnchor + 3-turn window + session_facts. No drift-score measurement (KPI-18)."),
    ("Gates 6 & 7 — triage & subagents", "🟡 Partial", "Rule-based triage (placeholder for distilled classifier, KPI-04 unmeasured). Subagents are grants inside one process, not isolated contexts."),
    ("Gate 9 — design patterns", "🟡 Partial", "Breaker, abstention, HITL, fail-closed paths, audit WAL. No bulkhead pools, no admission control."),
    ("Gates 10 & 11 — observability, incidents", "🟡 Partial", "Required spans + in-memory ring + opt-in OTLP. No collector/Grafana/alerts."),
    ("Gate 12 — 36 KPIs", "🟡 Partial", "Scorecard (Tab 6): 36 parsed from design.md; most NOT MEASURED."),
    ("Gates 13 & 14 — governance, red team", "🟡 Partial", "PII + injection heuristics, RLS, hash chain, HITL. No signed MCP manifests; red-team suite is a small smoke set."),
    ("Gates 15 & 16 — scale, 50K load", "⬜ Not run", "No k6/JMeter run; 50K VUs infeasible on this host."),
    ("Gate 17 — Obsidian + Graphify", "🟡 Partial", "Wikilink graph, depth ≤2, token budget. No import/call edges; tiny vault."),
    ("Gate 18 — workbench", "🟡 Partial", "This app."),
]


def render():
    st.subheader("Architecture & workflow")
    st.markdown("**Contract checklist** (status is maintained by hand in `workbench/tab1_architecture.py`)")
    st.dataframe([{"Gate": g, "Status": s, "Notes": n} for g, s, n in GATES], hide_index=True, width="stretch")
    img = ROOT.parent / "docs" / "workflow_orchestration.png"
    if img.exists():
        with st.expander("workflow_orchestration.png", expanded=False):
            st.image(str(img))

    st.divider()
    st.markdown("**Live triage console** — the rule-based classifier runs locally; send to the API to see the full pipeline.")
    msg = st.text_input("Message", "Please approve the refund for order 123", key="t1_msg")
    t = triage(msg)
    c = st.columns(5)
    for col, (k, v) in zip(c, t.model_dump().items()):
        col.metric(k, v)
    if st.button("Send to API", key="t1_send"):
        r = api().post("/api/v1/agent/query", role="user", json={"message": msg})
        st.session_state["t1_resp"] = {"status": r.status_code, "body": r.json()}
    resp = st.session_state.get("t1_resp")
    if resp:
        body = resp["body"]
        if resp["status"] != 200:
            st.warning(f"Blocked / error — HTTP {resp['status']}: {body.get('name')} ({body.get('code')}) {body.get('message', '')}")
        else:
            st.success(body["answer"][:400]) if not body["abstained"] else st.info(body["answer"])
            st.caption(f"route={body['route']} · abstained={body['abstained']} · faithfulness={body['faithfulness']} · "
                       f"halt={body['halt_reason']} · ticket={body['hitl_ticket_id']} · trace={body['trace_id']}")
            tr = need_dev(api().get(f"/api/v1/dev/traces/{body['trace_id']}"))
            if tr:
                st.markdown("**ReAct step trace**")
                steps = [s for s in tr["spans"] if s["name"] == "agent.react.step"]
                if not steps:
                    st.caption("No ReAct steps (route skipped the loop).")
                for i, s in enumerate(steps, 1):
                    err = s["attrs"].get("error", 0)
                    with st.expander(f"Step {i}: {s['attrs'].get('tool')} — {s['ms']} ms" + (f" — error {err}" if err else "")):
                        st.json(s["attrs"])

    st.divider()
    st.markdown("**Tier 3 HITL queue** (reviewer role; a requester can't approve their own ticket — use a different reviewer id)")
    rid = st.text_input("Reviewer user id", "reviewer-1", key="t1_rev")
    r = api().http.get("/api/v1/hitl/pending", headers=api().headers("hitl_reviewer", user=rid))
    if r.status_code != 200:
        st.error(f"HTTP {r.status_code}: {r.text[:200]}")
        return
    tickets = r.json()["tickets"]
    if not tickets:
        st.caption("No pending tickets for this tenant.")
    for tk in tickets[:20]:
        a, b, c2 = st.columns([6, 1, 1])
        a.write(f"`{tk['ticket_id']}` · {tk['workflow']} · requested {tk['requested_at'][:19]}")
        for col, d in ((b, "approve"), (c2, "reject")):
            if col.button(d.title(), key=f"{d}-{tk['ticket_id']}"):
                rr = api().http.post(f"/api/v1/hitl/{tk['ticket_id']}/decision", json={"decision": d}, headers=api().headers("hitl_reviewer", user=rid))
                st.toast(f"{rr.status_code}: {rr.json().get('status') or rr.json().get('message')}")
                st.rerun()
    st.caption("Approving records the decision only — no workflow executor exists (executed: false).")
