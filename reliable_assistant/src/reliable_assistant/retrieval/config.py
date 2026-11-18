"""Model selection. Doc models (design.md §1.2.2/3) are the production target; the dev defaults are the
small models already cached on this CPU-only host. Override with env vars to use the doc models."""
import os

# Production target:  BAAI/bge-large-en-v1.5  and  BAAI/bge-reranker-v2-m3
EMBED_MODEL = os.environ.get("RA_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.environ.get("RA_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_TOKENS, CHUNK_OVERLAP = 512, 64
HNSW_M, HNSW_EF_SEARCH = 32, 64
