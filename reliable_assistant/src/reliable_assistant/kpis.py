"""36-KPI release scorecard (design.md §1.5). Thresholds are parsed from design.md so the two cannot drift.
Statuses:  PASS / FAIL            measured on a sample the rule considers adequate (on THIS hardware)
           INDICATIVE ✓ / ✗       measured, but the sample is too small to count as evidence for the gate
           NOT MEASURED           no measurement exists (reason given) — never shown as a pass"""
import re
from pathlib import Path

DESIGN = Path(__file__).resolve().parents[3] / "design.md"
_ROW = re.compile(r"^\| (\d{2}) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", re.M)

# id -> (metric key, comparator, limit, min_n for a valid PASS/FAIL (None => always indicative))
RULES = {
    1: ("span.hook.pre.p95", "<=", 15, 20), 5: ("span.triage.classify.p95", "<=", 25, 20),
    7: ("span.mcp.rag.bm25.p95", "<=", 12, 20), 8: ("span.mcp.rag.biencoder.p95", "<=", 25, 20),
    9: ("span.mcp.rag.rerank.p95", "<=", 45, 20),
    14: ("harness.first_pass_pct", ">=", 96.0, 30), 15: ("harness.repair_pct", ">=", 90.0, 30),
    16: ("chaos.escape_pct", ">=", 100.0, 1), 17: ("audit.steps_mean", "<=", 2.2, 10),
    19: ("audit.hitl_pct", ">=", 100.0, 3), 20: ("audit.faith_mean", ">=", 0.92, 30),
    2: (("eval.pii_pct", "rt.pii_pct"), ">=", 99.8, 1000), 3: (("eval.injection_pct", "rt.injection_pct"), ">=", 99.5, 1000),
    6: (("eval.false_positive_pct", "rt.false_positive_pct"), "<=", 1.0, 500), 4: ("eval.triage_acc_pct", ">=", 97.0, 200),
    10: ("eval.recall50", ">=", 0.92, 100), 11: ("eval.precision4", ">=", 0.88, 100), 12: ("eval.mrr4", ">=", 0.85, 100),
    13: ("eval.ndcg4", ">=", 0.86, 100), 23: (("eval.abstention_pct", "rt.abstention_pct"), ">=", 98.0, 100), 26: ("audit.token_reduction_pct", ">=", 45.0, None),
    27: ("audit.cache_hit_pct", ">=", 35.0, None), 28: ("rt.cross_tenant_leaks", "<=", 0, None),
}
NOT_MEASURED = {
    18: "embedding drift audit not built",
    21: "RAGAS answer relevancy not run (lexical proxy only)", 22: "RAGAS entity recall not run",
    24: "needs NLI claim verifier at scale", 25: "no SSE /stream endpoint yet", 29: "no token-cost accounting",
    30: "MCP serialization overhead not isolated from server time",
    31: "defined @50K VUs; k6/JMeter not run", 32: "defined @50K VUs; k6/JMeter not run", 33: "defined @50K VUs; k6/JMeter not run",
    34: "defined @50K VUs; k6/JMeter not run", 35: "defined @50K VUs; k6/JMeter not run", 36: "needs a 1-hr soak run",
}


def load_kpis(path: Path = DESIGN) -> list[dict]:
    text = path.read_text()
    text = text[text.index("## 1.5"):text.index("## 1.6")]      # only the KPI matrix
    return [{"id": int(i), "stage": s.strip(), "name": n.strip(), "threshold": t.strip(), "tool": tool.strip()}
            for i, s, n, t, tool in _ROW.findall(text)]


def _cmp(v, op, lim):
    return v <= lim if op == "<=" else v >= lim


def evaluate(metrics: dict) -> list[dict]:
    rows = []
    for k in load_kpis():
        i, status, value, note = k["id"], "NOT MEASURED", None, NOT_MEASURED.get(k["id"], "no measurement source")
        if i in RULES:
            key, op, lim, min_n = RULES[i]
            m = next((metrics[k] for k in ((key,) if isinstance(key, str) else key) if k in metrics), None)
            if m is None or m.get("value") is None:
                note = "no data yet — run traffic / the red-team suite"
            else:
                value, n = m["value"], m.get("n", 0)
                ok = _cmp(value, op, lim)
                # eval-derived metrics come from author-generated corpora (independent=False): never certifying
                valid = min_n is not None and n >= min_n and m.get("independent", True)
                status = ("PASS" if ok else "FAIL") if valid else ("INDICATIVE ✓" if ok else "INDICATIVE ✗")
                note = f"n={n}" + ("" if valid else (" (sample too small to certify)" if m.get("independent", True) else " (author-generated eval set: not independent evidence)")) + f" · {m.get('note', '')}".rstrip(" ·")
        rows.append({**k, "value": value, "status": status, "note": note})
    return rows
