"""Tenant-scoped hybrid index: BM25Okapi+Porter, FAISS HNSW, RRF. One index per tenant => no cross-tenant path."""
import hashlib
import logging
import math
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from ..plugins.injection import is_injection
from . import config

log = logging.getLogger(__name__)
_stem = PorterStemmer()
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [_stem.stem(t) for t in _TOKEN.findall(text.lower())]


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Unweighted Reciprocal Rank Fusion: score = sum 1/(k + rank), rank 1-based (ADR-1)."""
    out: dict[str, float] = {}
    for ranking in rankings:
        for i, cid in enumerate(ranking, start=1):
            out[cid] = out.get(cid, 0.0) + 1.0 / (k + i)
    return out


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    chunk_hash: str
    text: str
    source_uri: str


def chunk_text(text: str, tokenizer, size: int = config.CHUNK_TOKENS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= size:
        return [text.strip()] if text.strip() else []
    out, step = [], size - overlap
    for start in range(0, len(ids), step):
        out.append(tokenizer.decode(ids[start:start + size]).strip())
        if start + size >= len(ids):
            break
    return out


class TenantIndex:
    def __init__(self, tenant_id: str, chunks: list[Chunk], embedder):
        self.tenant_id = tenant_id
        self.acl_fp = hashlib.sha256(tenant_id.encode()).hexdigest()[:16]
        self.embedder = embedder
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self._lock = threading.Lock()
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None
        self.faiss = None
        if chunks:
            vecs = self._embed([c.text for c in chunks])
            self.faiss = faiss.IndexHNSWFlat(vecs.shape[1], config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
            self.faiss.hnsw.efSearch = config.HNSW_EF_SEARCH
            self.faiss.add(vecs)

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.embedder.encode(texts, normalize_embeddings=True, batch_size=32), dtype="float32")

    def search(self, query: str, sparse_top_k=50, dense_top_k=50, k1=1.25, b=0.75, rrf_k=60, pool=40) -> dict:
        if not self.chunks:
            return {"chunks": [], "bm25_ms": 0.0, "dense_ms": 0.0, "rrf_ms": 0.0, "degraded": "none"}
        t0 = time.perf_counter()
        with self._lock:  # rank_bm25 reads k1/b at scoring time
            self.bm25.k1, self.bm25.b = k1, b
            scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:sparse_top_k]
        sparse = [(self.chunks[i].chunk_id, float(scores[i])) for i in order if scores[i] > 0]
        t1 = time.perf_counter()
        qv = self._embed([config.BGE_QUERY_PREFIX + query])
        sims, idx = self.faiss.search(qv, min(dense_top_k, len(self.chunks)))
        dense = [(self.chunks[i].chunk_id, float(s)) for s, i in zip(sims[0], idx[0]) if i >= 0]
        t2 = time.perf_counter()
        fused = rrf([[c for c, _ in sparse], [c for c, _ in dense]], k=rrf_k)
        bm, cos = dict(sparse), dict(dense)
        best: dict[str, tuple[str, float]] = {}  # dedup by chunk_hash, keep highest RRF
        for cid, sc in fused.items():
            h = self.by_id[cid].chunk_hash
            if h not in best or sc > best[h][1]:
                best[h] = (cid, sc)
        top = sorted(best.values(), key=lambda x: -x[1])[:pool]
        t3 = time.perf_counter()
        out = [{"chunk_id": cid, "chunk_hash": self.by_id[cid].chunk_hash, "text": self.by_id[cid].text[:3000],
                "bm25_score": bm.get(cid), "cosine_score": cos.get(cid), "rrf_score": sc,
                "source_uri": self.by_id[cid].source_uri, "acl_fingerprint": self.acl_fp} for cid, sc in top]
        return {"chunks": out, "bm25_ms": (t1 - t0) * 1e3, "dense_ms": (t2 - t1) * 1e3,
                "rrf_ms": (t3 - t2) * 1e3, "degraded": "none"}


_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)


def strip_frontmatter(md: str) -> str:
    return _FRONTMATTER.sub("", md, count=1)


def load_tenant(tenant_id: str, vault: Path, embedder) -> TenantIndex:
    chunks: list[Chunk] = []
    root = (vault / tenant_id).resolve()
    if root.is_dir() and root.parent == vault.resolve():
        for md in sorted(root.glob("*.md")):
            for n, text in enumerate(chunk_text(strip_frontmatter(md.read_text()), embedder.tokenizer)):
                if is_injection(text):  # indirect injection: quarantine at ingestion, never index
                    log.warning("quarantined chunk %s:%s:%d (injection)", tenant_id, md.stem, n)
                    continue
                chunks.append(Chunk(f"{tenant_id}:{md.stem}:{n}", hashlib.sha256(text.encode()).hexdigest(),
                                    text, f"rag://chunks/{tenant_id}:{md.stem}:{n}"))
    return TenantIndex(tenant_id, chunks, embedder)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
