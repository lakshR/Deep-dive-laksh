import pandas as pd
import streamlit as st

from reliable_assistant import redteam

from .common import api, need_dev

TOOLS = ["hybrid_retrieve", "rerank", "graphify_expand", "validate_sku", "validate_lot", "workflow_submit"]


def render():
    st.subheader("Red team & circuit breaker")
    st.caption("Black-box: drives the live API as a client. Sample sizes are small — these are smoke checks, not KPI measurements.")
    if st.button("▶ Run adversarial suite", type="primary"):
        with st.spinner("Running… (LLM-path questions can take a while on the free-tier key)"):
            rows = redteam.run_suite(api().http, lambda t, u, r: api().headers(r, t, u), tenants=(api().tenant, "globex" if api().tenant != "globex" else "acme"))
        st.session_state["rt_rows"] = rows
    rows = st.session_state.get("rt_rows")
    if rows:
        summ = redteam.summarize(rows)
        cols = st.columns(len(summ))
        for col, (cat, v) in zip(cols, summ.items()):
            col.metric(cat.replace("_", " "), f"{v['passed']}/{v['n']}")
            col.progress(v["passed"] / v["n"])
        df = pd.DataFrame(rows)
        df["result"] = df["passed"].map({True: "✅ pass", False: "❌ FAIL"})
        st.dataframe(df[["category", "name", "expected", "result", "detail"]], hide_index=True, width="stretch")
        bad = df[~df["passed"]]
        if len(bad):
            st.error(f"{len(bad)} case(s) failed — these are real gaps, not test noise.")

    st.divider()
    st.markdown("**Circuit breaker**")
    b = need_dev(api().get("/api/v1/dev/breaker"))
    if not b:
        return
    st.write(f"Auto-trip: the same (tool, canonical args) more than **{b['max_identical_calls']}×** in one request → `HALTED_BREAKER`.")
    st.write("Manually tripped tools: " + (", ".join(f"`{t}`" for t in b["forced_open"]) or "none"))
    c = st.columns([3, 1, 1])
    tool = c[0].selectbox("Tool", TOOLS, label_visibility="collapsed")
    if c[1].button("Trip"):
        api().post("/api/v1/dev/breaker/trip", json={"tool": tool}); st.rerun()
    if c[2].button("Reset all"):
        api().post("/api/v1/dev/breaker/reset"); st.rerun()
    if st.button("Run chaos loop: retry the 'same' SKU forever, alternating SKU99 / SKU-099"):
        j = need_dev(api().post("/api/v1/dev/chaos/loop", json={"mode": "alternating_sku"}))
        if j:
            (st.success if j["state"] == "HALTED_BREAKER" else st.error)(f"{j['state']} after {j['attempts']} attempts, {j['executed']} actually executed")
            st.dataframe(pd.DataFrame(j["steps"]), hide_index=True)
    if b["events"]:
        st.markdown("Recent breaker denials")
        st.dataframe(pd.DataFrame(b["events"]), hide_index=True, width="stretch")
