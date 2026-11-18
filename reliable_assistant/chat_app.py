"""Reliable Assistant — simple end-user app: Chat, Approvals (reviewers), Audit (auditors).
Run: streamlit run chat_app.py --server.port 8502
DEV SIGN-IN: there is no real login yet (OIDC is not built). This app mints its own dev JWTs — never expose it."""
import sys
from pathlib import Path

import httpx
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from reliable_assistant.auth import mint_token  # noqa: E402

st.set_page_config(page_title="Reliable Assistant", page_icon="💬")
API = st.sidebar.text_input("API URL", "http://localhost:8000")
st.sidebar.subheader("Dev sign-in")
tenant = st.sidebar.text_input("Organization (tenant)", "acme")
user = st.sidebar.text_input("User id", "alice")
role = st.sidebar.selectbox("Role", ["user", "hitl_reviewer", "auditor"])
if st.sidebar.button("New conversation"):
    st.session_state.pop("history", None); st.session_state.pop("sid", None)
st.sidebar.caption("Dev sign-in only: real login (OIDC) is not built.")
H = {"Authorization": f"Bearer {mint_token(tenant, user, role)}"}
http = httpx.Client(base_url=API, timeout=120)

pages = ["Chat"] + (["Approvals"] if role == "hitl_reviewer" else []) + (["Audit"] if role == "auditor" else [])
tabs = st.tabs(pages)

with tabs[0]:
    st.title("Ask your documents")
    hist = st.session_state.setdefault("history", [])
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
            if m.get("sources"):
                with st.expander(f"Sources ({len(m['sources'])})"):
                    for s in m["sources"]:
                        st.code(s, language=None)
    if q := st.chat_input("Ask a question about your organization's documents…"):
        hist.append({"role": "user", "text": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"), st.spinner("Searching your documents…"):
            try:
                r = http.post("/api/v1/agent/query", headers=H, json={"message": q, "session_id": st.session_state.get("sid")})
            except httpx.HTTPError:
                st.error("The service is not reachable."); st.stop()
            j = r.json()
            if r.status_code == 200:
                st.session_state["sid"] = j.get("session_id")
                if j["hitl_ticket_id"]:
                    text = f"⏳ **Needs human approval.** Ticket `{j['hitl_ticket_id']}` is pending. No action has been taken."
                elif j["abstained"]:
                    text = f"🤷 {j['answer']}"
                else:
                    text = j["answer"] + (f"\n\n<sub>faithfulness {j['faithfulness']:.2f}</sub>" if j["faithfulness"] is not None else "")
                st.markdown(text, unsafe_allow_html=True)
                src = [c["chunk_id"] for c in j["citations"]]
                if src:
                    with st.expander(f"Sources ({len(src)})"):
                        for s in src:
                            st.code(s, language=None)
                hist.append({"role": "assistant", "text": text, "sources": src})
            elif r.status_code == 422 and j.get("code") in (-32014, -32013):
                msg = "🛡️ That message was blocked by a safety check. Please rephrase."
                st.warning(msg); hist.append({"role": "assistant", "text": msg})
            else:
                st.error(f"Something went wrong ({j.get('name', r.status_code)}).")

if "Approvals" in pages:
    with tabs[pages.index("Approvals")]:
        st.title("Pending approvals")
        r = http.get("/api/v1/hitl/pending", headers=H)
        if r.status_code != 200:
            st.error(r.text[:200])
        else:
            tickets = r.json()["tickets"]
            st.caption("You cannot approve a request you made yourself. Approving records the decision only; nothing is executed.")
            if not tickets:
                st.success("Nothing waiting.")
            for t in tickets:
                c = st.columns([6, 1, 1])
                c[0].write(f"**{t['workflow'].replace('_', ' ')}** · `{t['ticket_id']}` · {t['requested_at'][:16]}")
                for col, d in ((c[1], "approve"), (c[2], "reject")):
                    if col.button(d.title(), key=f"{d}{t['ticket_id']}"):
                        rr = http.post(f"/api/v1/hitl/{t['ticket_id']}/decision", headers=H, json={"decision": d})
                        st.toast("Recorded." if rr.status_code == 200 else rr.json().get("message", "Failed"))
                        st.rerun()

if "Audit" in pages:
    with tabs[pages.index("Audit")]:
        st.title("Audit log")
        r = http.get("/api/v1/audit/recent", headers=H, params={"limit": 100})
        if r.status_code != 200:
            st.error(r.text[:200])
        else:
            j = r.json()
            ch = j["chain"]
            (st.success if ch["ok"] else st.error)(f"Hash chain {'verified' if ch['ok'] else 'BROKEN at seq ' + str(ch['first_bad_seq'])} · {ch['records']} records")
            rows = []
            for x in j["records"]:
                p = x["payload"]; d = {k: v for dd in p.get("plugin_decisions", []) for k, v in dd.items()}
                rows.append({"seq": x["seq"], "time": x["created_at"][:19], "query (redacted)": p["query_redacted"][:80],
                             "outcome": d.get("state") or d.get("blocked") or "", "faithfulness": p.get("faithfulness"),
                             "evidence": len(p.get("evidence_ids", []))})
            st.dataframe(rows, hide_index=True, width="stretch")
            st.caption("Only redacted text is stored. Records are append-only and hash-chained.")
