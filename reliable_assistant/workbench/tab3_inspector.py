import pandas as pd
import streamlit as st

from .common import api, need_dev

SECURITY = {"pii_redactor", "injection_shield", "hitl", "circuit_breaker"}


def render():
    st.subheader("MCP & plugin inspector")
    s = need_dev(api().get("/api/v1/dev/status"))
    if not s:
        return
    cols = st.columns(3)
    for col, m in zip(cols, s["mcp"]):
        with col:
            st.metric(m["server"], "● connected" if m["connected"] else "○ DOWN")
            st.caption(", ".join(m["tools"]) or "—")
    c = st.columns(2)
    c[0].metric("LLM", "● configured" if s["llm"] else "○ extractive fallback"); c[1].metric("Answer cache", "● on" if s["cache"] else "○ off")

    st.markdown("**Plugins** — hot toggles apply to live traffic and are audited. "
                "Turning off a ⚠ security plugin disables that control for every request until re-enabled.")
    for p in sorted(s["plugins"], key=lambda p: (p["stage"], p["priority"] or 0)):
        a, b = st.columns([5, 1])
        a.write(f"{'⚠ ' if p['name'] in SECURITY else ''}`{p['name']}` · {p['stage']} · priority {p['priority']} · "
                f"{'fail-closed' if p['fail_closed'] else 'fail-open'}")
        new = b.toggle("on", value=p["enabled"], key=f"tg-{p['name']}", label_visibility="collapsed")
        if new != p["enabled"]:
            r = api().post(f"/api/v1/dev/plugins/{p['name']}", json={"enabled": new})
            st.toast(f"{p['name']} → {'enabled' if new else 'DISABLED'} ({r.status_code})")
            st.rerun()

    st.divider()
    st.markdown("**JSON-RPC payload inspector** (most recent MCP calls from the API process)")
    st.button("Refresh", key="t3_ref")
    j = need_dev(api().get("/api/v1/dev/jsonrpc"))
    if not j or not j["calls"]:
        st.caption("No MCP calls recorded yet — send a query.")
        return
    calls = j["calls"]
    st.dataframe(pd.DataFrame([{"server": c["server"], "tool": c["tool"], "ms": c["ms"], "error": c.get("error", "")} for c in calls]),
                 hide_index=True, width="stretch")
    i = st.selectbox("Inspect call", range(len(calls)), format_func=lambda i: f"{i}: {calls[i]['server']}.{calls[i]['tool']} ({calls[i]['ms']} ms)")
    x, y = st.columns(2)
    x.caption("request"); x.code(calls[i]["request"], language="json")
    y.caption("response / error"); y.code(calls[i].get("response") or calls[i].get("error", ""), language="json")
