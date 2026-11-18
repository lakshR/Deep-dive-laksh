import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import altair as alt
import pandas as pd
import streamlit as st

from reliable_assistant import redteam
from reliable_assistant.kpis import evaluate

from .common import BLUE, GRAY, RED, STATUS_ICON, api, need_dev, themed

SLA_LO, SLA_HI = 200, 600


def waterfall():
    st.markdown("**Live span waterfall** (OpenTelemetry spans recorded for this tenant)")
    tr = need_dev(api().get("/api/v1/dev/traces"))
    if not tr or not tr["traces"]:
        st.caption("No traces yet — send a query from Tab 1.")
        return
    tid = st.selectbox("Trace", [t["trace_id"] for t in tr["traces"]], format_func=lambda x: f"{x[:12]}…  {next(t['ms'] for t in tr['traces'] if t['trace_id'] == x)} ms")
    sp = need_dev(api().get(f"/api/v1/dev/traces/{tid}"))
    if not sp:
        return
    df = pd.DataFrame(sp["spans"])
    t0 = df["start_ns"].min()
    df["start_ms"] = (df["start_ns"] - t0) / 1e6
    df["end_ms"] = df["start_ms"] + df["ms"].clip(lower=0.05)
    df["error"] = df["attrs"].apply(lambda a: bool(a.get("error")))
    order = list(df["name"]) + []
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=3, size=12).encode(
        y=alt.Y("name:N", sort=order, title=None), x=alt.X("start_ms:Q", title="ms since request start"), x2="end_ms:Q",
        color=alt.Color("error:N", scale=alt.Scale(domain=[False, True], range=[BLUE, RED]), legend=alt.Legend(title="span errored", orient="bottom")),
        tooltip=["name", alt.Tooltip("ms:Q", format=".2f"), alt.Tooltip("start_ms:Q", format=".1f")])
    st.altair_chart(themed(ch.properties(height=max(160, 26 * len(df)))), width="stretch")


def meters():
    st.markdown("**RAGAS triad** — real RAGAS is not run (needs several LLM calls per answer); what exists is below")
    res = need_dev(api().get("/api/v1/dev/metrics"))
    m = res["metrics"] if res else {}
    c = st.columns(3)
    f = m.get("audit.faith_mean")
    c[0].metric("Faithfulness (claim-verifier, mean)", f"{f['value']:.2f}" if f else "n/a", help="Independent-model claim check, gate ≥ 0.92 (KPI-20)")
    if f: c[0].progress(min(1.0, f["value"]))
    c[1].metric("Answer relevancy", "not measured"); c[2].metric("Context entity recall", "not measured")
    return m


def load_section():
    st.markdown("**Latency histogram vs 200–600 ms SLA**")
    st.warning("The 50,000-VU k6/JMeter run has **not** been performed (infeasible on this 4-CPU host). "
               "Below: a small dev smoke load, or a k6 summary you upload. KPI-31…36 stay NOT MEASURED.")
    a, b, c = st.columns(3)
    n = a.number_input("Requests", 5, 200, 30); conc = b.number_input("Concurrency", 1, 16, 4); uniq = c.checkbox("Bypass cache (unique queries)", False)
    if st.button("Run dev smoke load"):
        qs = redteam.BENIGN; cli = api()  # resolve in the script thread: session_state isn't readable from workers
        def one(i):
            q = qs[i % len(qs)] + (f" #{uuid.uuid4().hex[:6]}" if uniq else "")
            t = time.perf_counter()
            r = cli.post("/api/v1/agent/query", role="user", json={"message": q})
            return (time.perf_counter() - t) * 1e3, r.status_code
        with st.spinner("Running…"), ThreadPoolExecutor(conc) as ex:
            t0 = time.perf_counter(); res = list(ex.map(one, range(int(n)))); wall = time.perf_counter() - t0
        st.session_state["load"] = {"ms": [r[0] for r in res], "codes": [r[1] for r in res], "wall": wall, "conc": int(conc)}
    L = st.session_state.get("load")
    if L:
        ms = sorted(L["ms"]); q = lambda p: ms[min(len(ms) - 1, int(round(p * (len(ms) - 1))))]
        m = st.columns(5)
        m[0].metric("p50", f"{q(.5):.0f} ms"); m[1].metric("p95", f"{q(.95):.0f} ms"); m[2].metric("p99", f"{q(.99):.0f} ms")
        m[3].metric("Throughput", f"{len(ms) / L['wall']:.1f} RPS"); m[4].metric("Non-2xx", sum(1 for c in L["codes"] if c >= 400))
        df = pd.DataFrame({"ms": ms})
        band = alt.Chart(pd.DataFrame({"a": [SLA_LO], "b": [SLA_HI]})).mark_rect(opacity=0.12, color=BLUE).encode(x="a:Q", x2="b:Q")
        h = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=3).encode(x=alt.X("ms:Q", bin=alt.Bin(maxbins=30), title="end-to-end latency (ms)"), y=alt.Y("count():Q", title="requests"))
        rules = alt.Chart(pd.DataFrame({"x": [SLA_LO, SLA_HI], "l": ["200 ms", "600 ms SLA"]})).mark_rule(strokeDash=[4, 3], color="#8a8984").encode(x="x:Q")
        st.altair_chart(themed(alt.layer(band, h, rules).properties(height=240)), width="stretch")
        inside = sum(1 for v in ms if v <= SLA_HI) / len(ms)
        st.caption(f"{L['conc']} concurrent clients, {len(ms)} requests — {inside:.0%} finished within 600 ms. Shaded band = the 200–600 ms target. "
                   "This is not a 50K-VU result.")
    up = st.file_uploader("…or upload a k6 --summary-export JSON", type="json")
    if up:
        try:
            d = json.load(up)["metrics"]["http_req_duration"]; d = d.get("values", d)
            vals = {"p50": d.get("med", d.get("p(50)")), "p95": d.get("p(95)"), "p99": d.get("p(99)")}
            df = pd.DataFrame({"pct": list(vals), "ms": [v for v in vals.values()]}).dropna()
            ch = alt.Chart(df).mark_bar(color=BLUE, cornerRadiusEnd=4, size=28).encode(x=alt.X("pct:N", title=None, sort=None), y=alt.Y("ms:Q", title="ms"), tooltip=["pct", "ms"])
            rule = alt.Chart(pd.DataFrame({"y": [SLA_HI]})).mark_rule(strokeDash=[4, 3], color="#8a8984").encode(y="y:Q")
            st.altair_chart(themed(alt.layer(ch, rule).properties(height=220)), width="stretch")
            st.caption("p99 appears only if the run used `--summary-trend-stats` including p(99).")
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read k6 summary: {e}")


def rt_metrics() -> dict:
    rows = st.session_state.get("rt_rows")
    if not rows:
        return {}
    s = redteam.summarize(rows); g = lambda c: s.get(c, {"n": 0, "passed": 0})
    pct = lambda p, n: 100.0 * p / n if n else None
    inj = g("injection_direct")["passed"] + g("injection_indirect")["passed"], g("injection_direct")["n"] + g("injection_indirect")["n"]
    out = {"rt.pii_pct": {"value": pct(g("pii_obfuscated")["passed"], g("pii_obfuscated")["n"]), "n": g("pii_obfuscated")["n"]},
           "rt.injection_pct": {"value": pct(*inj), "n": inj[1]},
           "rt.false_positive_pct": {"value": 100 - pct(g("benign_control")["passed"], g("benign_control")["n"]), "n": g("benign_control")["n"]},
           "rt.abstention_pct": {"value": pct(g("abstention")["passed"], g("abstention")["n"]), "n": g("abstention")["n"]},
           "rt.cross_tenant_leaks": {"value": 0 if g("cross_tenant")["passed"] else 1, "n": g("cross_tenant")["n"]}}
    return {k: v for k, v in out.items() if v["value"] is not None}


def render():
    st.subheader("36-KPI scorecard & load")
    waterfall()
    st.divider()
    m = meters()
    st.divider()
    load_section()
    st.divider()
    st.markdown("**Release scorecard** (thresholds parsed from `design.md` §1.5)")
    mm = dict(m); mm.update(rt_metrics())
    if "rt_rows" not in st.session_state:
        st.info("Run the adversarial suite (Tab 5) to populate the red-team KPIs; send queries to populate latency/loop KPIs.")
    rows = evaluate(mm)
    cnt = {k: sum(1 for r in rows if r["status"] == k) for k in STATUS_ICON}
    ready = cnt["FAIL"] == 0 and cnt["NOT MEASURED"] == 0 and cnt["INDICATIVE ✗"] == 0 and cnt["INDICATIVE ✓"] == 0
    (st.success if ready else st.error)(
        f"{'RELEASE GATE: PASS' if ready else 'RELEASE GATE: NOT READY'} — {cnt['PASS']} pass · {cnt['FAIL']} fail · "
        f"{cnt['INDICATIVE ✓'] + cnt['INDICATIVE ✗']} indicative · {cnt['NOT MEASURED']} not measured (all 36 must pass)")
    df = pd.DataFrame([{"#": r["id"], "Stage": r["stage"], "KPI": r["name"], "Gate": r["threshold"],
                        "Measured": "—" if r["value"] is None else str(r["value"]), "Status": STATUS_ICON[r["status"]], "Note": r["note"]} for r in rows])
    f = st.multiselect("Filter status", list(STATUS_ICON.values()), default=[])
    st.dataframe(df[df["Status"].isin(f)] if f else df, hide_index=True, width="stretch", height=620)
    st.caption("Latency KPIs are measured on this dev host (CPU, FP32, rate-limited LLM) — failing them here says nothing about the target hardware, "
               "and passing them says nothing about 50K VUs.")
