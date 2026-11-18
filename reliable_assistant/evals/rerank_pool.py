"""How much does reranking fewer candidates buy on CPU, and what does it cost in quality? (design.md uses pool=40.)"""
import math, statistics, sys, time, uuid
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import corpora
from reliable_assistant.retrieval.service import RetrievalService
from reliable_assistant.types import HybridRetrieveIn, RerankIn
docs, queries, _ = corpora.retrieval_corpus()
vault = HERE / 'vault' / 'evalco'; vault.mkdir(parents=True, exist_ok=True)
[f.unlink() for f in vault.glob('*.md')]
[(vault / f'{stem}.md').write_text(body) for stem, body in docs]
svc = RetrievalService(HERE / "vault")
meta = lambda: dict(tenant_id="evalco", user_id_hash="a" * 32, request_id="0" * 26, trace_id="b" * 32, deadline_ms=2000, idempotency_key=str(uuid.uuid4()))
sample = queries[::3][:60]
pools = [HybridRetrieveIn(meta=meta(), query=q["q"]) for q in sample]
cands = [svc.hybrid_retrieve(p).chunks for p in pools]
print(f"{'pool':>5} {'rerank p50':>11} {'p95':>8} {'MRR@4':>7} {'gold kept':>10} {'P@4':>6}")
for n in (5, 10, 20, 40):
    lat, rr, kept_hit, prec = [], [], 0, []
    for q, ch in zip(sample, cands):
        c = ch[:n]
        t = time.perf_counter(); r = svc.rerank(RerankIn(meta=meta(), query=q["q"], candidates=c)); lat.append((time.perf_counter() - t) * 1e3)
        order = [x for x, _ in r.all_scores]; rank = order.index(q["gold"]) + 1 if q["gold"] in order else 99
        rr.append(1 / rank if rank <= 4 else 0); k = [x for x, _ in r.ranked]
        kept_hit += q["gold"] in k; prec.append((q["gold"] in k) / len(k) if k else 0)
    lat.sort()
    print(f"{n:>5} {statistics.median(lat):>9.0f}ms {lat[int(.95 * (len(lat) - 1))]:>6.0f}ms {statistics.mean(rr):>7.3f} {kept_hit / len(sample):>9.0%} {statistics.mean(prec):>6.2f}")
