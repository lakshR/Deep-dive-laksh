import altair as alt
import pandas as pd
import streamlit as st

from .common import BLUE, GRAY, api, need_dev, themed


def render():
    st.subheader("3-stage hybrid RAG lab")
    q = st.text_input("Query", "How fast must LOT-88392 be quarantined?", key="lab_q")
    c1, c2, c3, c4 = st.columns(4)
    k1 = c1.slider("BM25 k1", 0.5, 2.5, 1.25, 0.05)
    b = c2.slider("BM25 b", 0.0, 1.0, 0.75, 0.05)
    rrf_k = c3.slider("RRF k", 1, 200, 60)
    thr = c4.slider("Abstain threshold", 0.0, 1.0, 0.42, 0.01)
    if not q.strip():
        return
    j = need_dev(api().post("/api/v1/dev/rag-lab", json={"query": q, "bm25_k1": k1, "bm25_b": b, "rrf_k": rrf_k, "threshold": thr}))
    if not j:
        return
    if not j["rows"]:
        st.warning("No candidates retrieved for this tenant.")
        return
    t = j["timing"]
    m = st.columns(4)
    m[0].metric("Stage 1 BM25", f"{t['bm25_ms']:.1f} ms"); m[1].metric("Stage 2 dense", f"{t['dense_ms']:.1f} ms")
    m[2].metric("RRF fuse", f"{t['rrf_ms']:.2f} ms"); m[3].metric("Top CE score", f"{j['top_score']:.3f}")
    (st.error if j["abstain"] else st.success)(
        f"{'ABSTAIN — top score below threshold: no LLM call, fixed refusal' if j['abstain'] else 'Grounded — answer would be generated'} "
        f"(top {j['top_score']:.3f} vs threshold {thr:.2f})")
    df = pd.DataFrame(j["rows"])
    st.dataframe(df.rename(columns={"bm25": "BM25 score", "bm25_rank": "BM25 rank", "cosine": "Bi-enc cosine", "dense_rank": "Dense rank",
                                    "rrf": "RRF score", "rrf_rank": "RRF rank", "cross_encoder": "CE score", "ce_rank": "CE rank"})
                 [["chunk_id", "BM25 score", "BM25 rank", "Bi-enc cosine", "Dense rank", "RRF score", "RRF rank", "CE score", "CE rank", "kept", "text"]],
                 hide_index=True, width="stretch")
    # one series: CE score by chunk. emphasis = kept (blue) vs below-threshold (gray); threshold as a labelled rule
    d = df.assign(state=df["kept"].map({True: "kept (above threshold)", False: "below threshold"}))
    bars = alt.Chart(d).mark_bar(cornerRadiusEnd=4, size=14).encode(
        y=alt.Y("chunk_id:N", sort="-x", title=None), x=alt.X("cross_encoder:Q", title="Cross-encoder score (sigmoid)", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("state:N", scale=alt.Scale(domain=["kept (above threshold)", "below threshold"], range=[BLUE, GRAY]), legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["chunk_id", alt.Tooltip("cross_encoder:Q", format=".3f"), "ce_rank", "bm25_rank", "dense_rank", "rrf_rank"])
    rule = alt.Chart(pd.DataFrame({"x": [thr], "label": [f"threshold {thr:.2f}"]})).mark_rule(strokeDash=[4, 3], color="#8a8984").encode(x="x:Q")
    txt = alt.Chart(pd.DataFrame({"x": [thr], "label": [f"threshold {thr:.2f}"]})).mark_text(align="left", dx=4, dy=-6, color="#8a8984", baseline="bottom").encode(x="x:Q", text="label:N", y=alt.value(0))
    st.altair_chart(themed(alt.layer(bars, rule, txt).properties(height=max(120, 36 * len(d)))), width="stretch")
    st.caption("Note: this host runs small CPU models; the 0.42 threshold is calibrated for bge-reranker-v2-m3, not MiniLM.")
