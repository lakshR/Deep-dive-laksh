import pandas as pd
import streamlit as st

from .common import BLUE, api, need_dev


def _dot(seeds, nodes) -> str:
    lines = ["digraph G { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica]; edge [color=\"#8a8984\"];"]
    for s in seeds:
        lines.append(f'"{s}" [style="rounded,filled", fillcolor="{BLUE}", fontcolor=white];')
    for n in nodes:
        lines.append(f'"{n["id"]}" [label="{n["id"]}\\nhop {n["hop"]} · v{n["version"]}"];')
        if n["edge_from"]:
            lines.append(f'"{n["edge_from"]}" -> "{n["id"]}";')
    return "\n".join(lines) + "}"


def render():
    st.subheader("Memory & Graphify tokens")
    notes = need_dev(api().get("/api/v1/dev/notes"))
    if not notes:
        return
    ns = notes["notes"]
    if not ns:
        st.warning("No notes in this tenant's vault.")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Graphify subgraph**")
            seed = st.selectbox("Seed note", [n["slug"] for n in ns], key="t4_seed")
            hops = st.radio("Hops", [1, 2], horizontal=True, index=1)
            edge = st.multiselect("Edge types", ["wikilink", "backlink"], default=["wikilink", "backlink"])
            g = need_dev(api().post("/api/v1/dev/graph", json={"seeds": [f"{api().tenant}:{seed}:0"], "hops": hops, "edge_types": edge or ["wikilink"]}))
            if g:
                st.graphviz_chart(_dot([seed], g["nodes"]), width="stretch")
                m = st.columns(2)
                m[0].metric("Tokens in subgraph", g["tokens_used"]); m[1].metric("Saved vs raw notes", g["tokens_saved_vs_raw"])
        with right:
            st.markdown("**Atomic note viewer**")
            slug = st.selectbox("Note", [n["slug"] for n in ns], key="t4_note")
            n = next(n for n in ns if n["slug"] == slug)
            st.json(n["meta"], expanded=False)
            st.caption("links: " + (", ".join(f"[[{l}]]" for l in n["links"]) or "—"))
            st.markdown(n["body"])

    st.divider()
    st.markdown("**Session state browser** (Redis hot window; facts live in Postgres `session_facts`)")
    ss = need_dev(api().get("/api/v1/dev/sessions"))
    if ss and ss["sessions"]:
        st.dataframe(pd.DataFrame([{"user:session": x["key"], "version": x["version"], "ttl_s": x["ttl_s"],
                                    "turn": (x["state"] or {}).get("turn_index"), "turns kept": len((x["state"] or {}).get("last_turns", []))} for x in ss["sessions"]]),
                     hide_index=True, width="stretch")
        pick = st.selectbox("Open session", range(len(ss["sessions"])), format_func=lambda i: ss["sessions"][i]["key"][-34:])
        st.json(ss["sessions"][pick]["state"], expanded=False)
    else:
        st.caption("No live sessions for this tenant (send a query with a session id).")

    st.divider()
    st.markdown("**Token savings** — evidence tokens actually sent to the LLM vs. dumping the whole retrieved pool")
    mets = need_dev(api().get("/api/v1/dev/metrics"))
    t = (mets or {}).get("metrics", {}).get("audit.token_reduction_pct")
    if not t:
        st.caption("No LLM-path queries recorded yet.")
        return
    c = st.columns(3)
    c[0].metric("Prompt-token reduction", f"{t['value']}%"); c[1].metric("Queries measured", t["n"]); c[2].metric("KPI-26 gate", "≥ 45 %")
    st.progress(min(1.0, max(0.0, t["value"] / 100)), text=f"{t['value']}% fewer prompt tokens than a raw dump of the retrieved pool")
    price = st.number_input("Assumed input price, $ per 1M tokens (your own number — not read from any provider)", 0.0, 100.0, 0.15, 0.05)
    st.caption(f"Indicative only: a {t['value']}% cut on a $ {price}/1M input price saves ${price * t['value'] / 100:.4f} per 1M raw-dump tokens. "
               "This is not KPI-29 (cost per 1,000 queries), which needs real token accounting. Tiny vault ⇒ not evidence for KPI-26.")
