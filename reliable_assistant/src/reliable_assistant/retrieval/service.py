"""Retrieval service: lazy per-tenant indexes + cross-encoder rerank with deterministic abstention (§1.2.3)."""
from pathlib import Path

from sentence_transformers import CrossEncoder, SentenceTransformer

from ..errors import AssistantError
from ..types import HybridRetrieveIn, HybridRetrieveOut, RerankIn, RerankOut
from . import config
from .index import TenantIndex, load_tenant, sigmoid


class RetrievalService:
    def __init__(self, vault: Path):
        self.vault = vault
        self.embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        self.reranker = CrossEncoder(config.RERANK_MODEL, device="cpu", max_length=512)
        self._indexes: dict[str, TenantIndex] = {}

    def index(self, tenant_id: str) -> TenantIndex:
        if tenant_id not in self._indexes:
            self._indexes[tenant_id] = load_tenant(tenant_id, self.vault, self.embedder)
        return self._indexes[tenant_id]

    def reload(self, tenant_id: str) -> None:
        self._indexes.pop(tenant_id, None)

    def hybrid_retrieve(self, p: HybridRetrieveIn) -> HybridRetrieveOut:
        res = self.index(p.meta.tenant_id).search(p.query, p.sparse_top_k, p.dense_top_k,
                                                  p.bm25_k1, p.bm25_b, p.rrf_k, p.pool)
        return HybridRetrieveOut(**res)

    def rerank(self, p: RerankIn) -> RerankOut:
        # ACL guard: every candidate must belong to the caller's tenant.
        fp = self.index(p.meta.tenant_id).acl_fp
        if any(c.acl_fingerprint != fp or not c.chunk_id.startswith(p.meta.tenant_id + ":") for c in p.candidates):
            raise AssistantError(-32012, "candidate outside tenant ACL")
        logits = self.reranker.predict([(p.query, c.text) for c in p.candidates], batch_size=32)
        scored = sorted(((c.chunk_id, sigmoid(float(s))) for c, s in zip(p.candidates, logits)),
                        key=lambda x: -x[1])
        top = scored[0][1]
        kept = [x for x in scored if x[1] > p.threshold][:p.rerank_k]
        return RerankOut(ranked=kept, top_score=top, abstain=top < p.threshold, precision="fp32", all_scores=scored)
