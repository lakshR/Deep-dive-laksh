"""MCP server `mcp-graphify-vault` (stdio). Tool: graphify_expand (read-only, tenant-scoped, depth ≤ 2)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp.server.mcpserver import MCPServer  # noqa: E402
from mcp.types import ToolAnnotations  # noqa: E402

from reliable_assistant.retrieval.graph import VaultGraph  # noqa: E402
from reliable_assistant.types import GraphifyExpandIn, GraphifyExpandOut  # noqa: E402

VAULT = Path(os.environ.get("RA_VAULT", ROOT / "knowledge_vault"))
_graphs: dict[str, VaultGraph] = {}
mcp = MCPServer("mcp-graphify-vault")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
          description="Expand seed notes/chunks over [[wikilinks]] (depth<=2) into minimal token-budgeted summaries.")
def graphify_expand(params: GraphifyExpandIn) -> GraphifyExpandOut:
    t = params.meta.tenant_id
    g = _graphs.setdefault(t, VaultGraph(t, VAULT))
    return GraphifyExpandOut(**g.expand(list(params.seeds), params.hops, params.token_budget, params.edge_types))


if __name__ == "__main__":
    mcp.run("stdio")
