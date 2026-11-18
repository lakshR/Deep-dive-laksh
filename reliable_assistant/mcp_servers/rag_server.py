"""MCP server `mcp-hybrid-rag` (stdio). Tools: hybrid_retrieve, rerank. Resource: rag://chunks/{id}.
Rejects calls without a valid _meta.tenant_id (-32011) via the Meta contract."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from reliable_assistant.retrieval.service import RetrievalService  # noqa: E402
from reliable_assistant.types import (HybridRetrieveIn, HybridRetrieveOut, RerankIn,  # noqa: E402
                                      RerankOut)

svc = RetrievalService(Path(os.environ.get("RA_VAULT", ROOT / "knowledge_vault")))
mcp = MCPServer("mcp-hybrid-rag")
RO = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


@mcp.tool(annotations=RO, description="Tenant-scoped BM25 + bi-encoder retrieval fused with RRF. Read-only.")
def hybrid_retrieve(params: HybridRetrieveIn) -> HybridRetrieveOut:
    return svc.hybrid_retrieve(params)


@mcp.tool(annotations=RO, description="Stage 3 cross-encoder rerank with deterministic abstention.")
def rerank(params: RerankIn) -> RerankOut:
    return svc.rerank(params)


if __name__ == "__main__":
    mcp.run("stdio")
