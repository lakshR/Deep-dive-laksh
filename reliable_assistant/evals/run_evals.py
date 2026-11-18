"""Offline evaluation runner. Usage: python evals/run_evals.py [--skip-retrieval]
Writes evals/results/latest.json, which the dev API folds into the KPI scorecard (capped at INDICATIVE: generated sets)."""
import json
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import corpora  # noqa: E402
from reliable_assistant.agent.triage import triage  # noqa: E402
from reliable_assistant.plugins import injection  # noqa: E402
from reliable_assistant.plugins.pii import redact  # noqa: E402

M, D = {}, {}


def put(key, value, n, note=""):
    M[key] = {"value": round(value, 4), "n": n, "note": note, "independent": False}


def pct(a, b): return 100.0 * a / b if b else 0.0
def p95(v): s = sorted(v); return s[min(len(s) - 1, int(round(.95 * (len(s) - 1))))]


def pii():
    cases, leaks, per = corpora.pii_cases(), [], {}
    for c in cases:
        out, counts = redact(c["text"])
        norm = re.sub(r"[​‌‍⁠﻿]", "", out).lower()
        digits = re.sub(r"\D", "", out)
        core = c["core"].lower()
        leaked = (core in digits) if core.isdigit() else (core in norm)
        ok = (not leaked) and bool(counts)
        p = per.setdefault(c["cat"], [0, 0]); p[0] += ok; p[1] += 1
        if not ok: leaks.append({"cat": c["cat"], "text": c["text"], "out": out})
    n, good = len(cases), sum(v[0] for v in per.values())
    put("eval.pii_generated_pct", pct(good, n), n, "generated corpus: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in per.items()))
    held, hfail = corpora.HELD_OUT_PII, []
    for text, core in held:
        out, counts = redact(text)
        if core.lower() in re.sub(r"[​‌‍⁠﻿]", "", out).lower() or core.lower() in re.sub(r"\D", "", out): hfail.append({"text": text, "out": out})
    put("eval.pii_pct", pct(len(held) - len(hfail), len(held)), len(held), "HELD-OUT hand-written PII in unusual forms (more realistic)")
    D["pii_missed_heldout"] = hfail
    D["pii_failures"] = leaks[:25]; D["pii_by_cat"] = per
    neg = corpora.non_pii_cases(); fp = [t for t in neg if redact(t)[1]]
    put("eval.pii_false_redaction_pct", pct(len(fp), len(neg)), len(neg), "non-PII ids/dates/versions redacted")
    D["pii_false_redactions"] = fp[:15]


def inj():
    atk = corpora.injection_attacks(); miss = [a for a in atk if not injection.is_injection(a)]
    held = corpora.HELD_OUT_ATTACKS; hmiss = [a for a in held if not injection.is_injection(a)]
    ben = corpora.benign_queries(); fp = [b for b in ben if injection.is_injection(b)]
    put("eval.injection_generated_pct", pct(len(atk) - len(miss), len(atk)), len(atk), "template-generated: shares the shield's author")
    put("eval.injection_pct", pct(len(held) - len(hmiss), len(held)), len(held), "HELD-OUT hand-written paraphrases (more realistic than the generated set)")
    put("eval.false_positive_pct", pct(len(fp), len(ben)), len(ben), "benign queries using trigger words (ignore/instructions/system/override…)")
    D["injection_missed_generated"] = miss[:15]; D["injection_missed_heldout"] = hmiss; D["injection_false_positives"] = fp[:15]


def promptguard_eval():
    from reliable_assistant.plugins import promptguard
    pg = promptguard.from_env()
    if pg is None:
        print("skip promptguard: set RA_INJECTION_MODEL and a key"); return
    held, ben = corpora.HELD_OUT_ATTACKS, corpora.benign_queries(100)
    sc = [pg.score(a) for a in held]; ok = [x for x in sc if x is not None]
    comb = sum(1 for a, x in zip(held, sc) if injection.is_injection(a) or (x is not None and x >= promptguard.THRESHOLD))
    bsc = [pg.score(b) for b in ben]; fp = sum(1 for b, x in zip(ben, bsc) if injection.is_injection(b) or (x is not None and x >= promptguard.THRESHOLD))
    put("eval.injection_promptguard_pct", pct(comb, len(held)), len(held), f"regex ∪ {pg.model} on held-out; {len(held) - len(ok)} calls unavailable")
    put("eval.injection_promptguard_fp_pct", pct(fp, len(ben)), len(ben), "benign false positives with classifier on")
    D["promptguard_missed"] = [a for a, x in zip(held, sc) if not injection.is_injection(a) and (x is None or x < promptguard.THRESHOLD)]


def tri():
    s, lat, wrong = corpora.triage_set(), [], []
    for q, lab in s:
        t0 = time.perf_counter(); got = triage(q).route; lat.append((time.perf_counter() - t0) * 1e3)
        if got != lab: wrong.append({"q": q, "expected": lab, "got": got})
    put("eval.triage_acc_pct", pct(len(s) - len(wrong), len(s)), len(s), "author-labelled set")
    put("eval.triage_p95_ms", p95(lat), len(s), "pure-python rules")
    D["triage_wrong"] = wrong


def retrieval():
    from reliable_assistant.retrieval.service import RetrievalService
    from reliable_assistant.types import HybridRetrieveIn, RerankIn
    docs, queries, unans = corpora.retrieval_corpus()
    vault = HERE / "vault" / "evalco"; vault.mkdir(parents=True, exist_ok=True)
    for f in vault.glob("*.md"): f.unlink()
    for stem, body in docs: (vault / f"{stem}.md").write_text(body)
    svc = RetrievalService(HERE / "vault")
    meta = lambda: dict(tenant_id="evalco", user_id_hash="a" * 32, request_id="0" * 26, trace_id="b" * 32, deadline_ms=2000, idempotency_key=str(uuid.uuid4()))
    hit50, rr_list, ndcg, prec, got_kept, bm, de, rrf_t, rer = 0, [], [], [], 0, [], [], [], []
    fails = []
    for q in queries:
        wide = svc.index("evalco").search(q["q"], pool=100)      # direct: the public contract caps the pool at 40
        hit50 += q["gold"] in {c["chunk_id"] for c in wide["chunks"]}
        pool = svc.hybrid_retrieve(HybridRetrieveIn(meta=meta(), query=q["q"]))
        bm.append(pool.bm25_ms); de.append(pool.dense_ms); rrf_t.append(pool.rrf_ms)
        t0 = time.perf_counter(); r = svc.rerank(RerankIn(meta=meta(), query=q["q"], candidates=pool.chunks)); rer.append((time.perf_counter() - t0) * 1e3)
        order = [c for c, _ in r.all_scores]; rank = order.index(q["gold"]) + 1 if q["gold"] in order else 99
        rr_list.append(1 / rank if rank <= 4 else 0.0); ndcg.append(1 / (1 + (rank - 1)).bit_length() if False else (1 / __import__("math").log2(rank + 1) if rank <= 4 else 0.0))
        kept = [c for c, _ in r.ranked]
        prec.append((q["gold"] in kept) / len(kept) if kept else 0.0); got_kept += q["gold"] in kept
        if rank > 1: fails.append({"q": q["q"], "gold": q["gold"], "rank": rank, "top": order[:2]})
    n = len(queries)
    put("eval.recall50_pct", pct(hit50, n), n, "gold doc in BM25-50 ∪ dense-50 union (pool=100), synthetic 60-doc corpus"); M["eval.recall50_pct"]["value"] /= 100
    put("eval.precision4", statistics.mean(prec), n, "precision over chunks kept after rerank (single relevant doc per query)")
    put("eval.mrr4", statistics.mean(rr_list), n, ""); put("eval.ndcg4", statistics.mean(ndcg), n, "binary relevance")
    put("eval.answer_in_kept_pct", pct(got_kept, n), n, "gold doc survives the 0.42 threshold")
    for k, v in (("bm25", bm), ("dense", de), ("rrf", rrf_t), ("rerank", rer)):
        put(f"eval.{k}_p95_ms", p95(v), n, "in-process, CPU, no MCP overhead")
    ab = sum(svc.rerank(RerankIn(meta=meta(), query=u, candidates=svc.hybrid_retrieve(HybridRetrieveIn(meta=meta(), query=u)).chunks)).abstain for u in unans)
    put("eval.abstention_pct", pct(ab, len(unans)), len(unans), "unanswerable: unknown products (hard negatives) + general knowledge")
    D["retrieval_rank_gt1"] = fails[:15]


if __name__ == "__main__":
    t = time.time()
    out = HERE / "results" / "latest.json"
    prev = json.loads(out.read_text()) if out.exists() and "--fresh" not in sys.argv else {"metrics": {}, "details": {}}
    M.update(prev["metrics"]); D.update(prev["details"])          # partial runs keep earlier results
    pii(); inj(); tri()
    if "--promptguard" in sys.argv: promptguard_eval()
    if "--retrieval" in sys.argv or not prev["metrics"]: retrieval()
    (HERE / "results").mkdir(exist_ok=True)
    out.write_text(json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": M, "details": D}, indent=1))
    print(f"done in {time.time() - t:.0f}s")
    for k, v in M.items(): print(f"{k:34s} {v['value']:>9}  n={v['n']:<5} {v['note'][:80]}")
