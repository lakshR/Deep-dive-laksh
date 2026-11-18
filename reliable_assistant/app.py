"""Reliable Assistant — developer workbench (Gate 18). Run:  streamlit run app.py
Talks to the API over HTTP (default http://localhost:8000, started with RA_DEV=1). Dev-only: it mints its own JWTs."""
import streamlit as st

from workbench import common, tab1_architecture, tab2_rag_lab, tab3_inspector, tab4_memory, tab5_redteam, tab6_scorecard

st.set_page_config(page_title="Reliable Assistant — Workbench", layout="wide")
with st.sidebar:
    st.title("Workbench")
    base = st.text_input("API URL", "http://localhost:8000")
    tenant = st.text_input("Tenant", "acme")
    user = st.text_input("User id", "workbench-user")
    st.session_state["api"] = common.Api(base, tenant, user)
    h = st.session_state["api"].alive()
    if h:
        st.success(f"API up · stage {h['stage']}")
        st.caption(f"state {'●' if h['state_store'] else '○'} · retrieval {'●' if h['retrieval'] else '○'} · llm {'●' if h['llm'] else '○'} · graph {'●' if h.get('graph') else '○'} · cache {'●' if h.get('cache') else '○'}")
    else:
        st.error("API not reachable")
    st.caption("Dev tool: mints its own JWTs with the dev secret. Never expose it, or the dev API, outside a trusted network.")

if not h:
    st.stop()
tabs = st.tabs(["1 · Architecture", "2 · RAG lab", "3 · MCP & plugins", "4 · Memory & Graphify", "5 · Red team", "6 · Scorecard & load"])
for tab, mod in zip(tabs, (tab1_architecture, tab2_rag_lab, tab3_inspector, tab4_memory, tab5_redteam, tab6_scorecard)):
    with tab:
        mod.render()
