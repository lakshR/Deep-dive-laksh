"""Tool registry: Stage 2 MCP tools + local Tier 2 validators + the Tier 3 workflow tool (never executes)."""
from ..errors import AssistantError
from ..types import (GraphifyExpandIn, GraphifyExpandOut, HybridRetrieveIn, HybridRetrieveOut, RerankIn, RerankOut, ValidateLotIn, ValidateSkuIn,
                     WorkflowSubmitIn, WorkflowSubmitOut)
from .harness import ToolSpec


def build_specs(rag, state=None, graph=None) -> list[ToolSpec]:
    async def hybrid(m): return (await rag.call("hybrid_retrieve", m.model_dump(mode="json"), HybridRetrieveOut)).model_dump(mode="json")
    async def rerank(m): return (await rag.call("rerank", m.model_dump(mode="json"), RerankOut)).model_dump(mode="json")
    async def graphify(m):
        if graph is None:
            raise AssistantError(-32002, "mcp-graphify-vault unavailable")
        return (await graph.call("graphify_expand", m.model_dump(mode="json"), GraphifyExpandOut)).model_dump(mode="json")
    async def vsku(m): return {"valid": True, "sku": m.sku}
    async def vlot(m): return {"valid": True, "lot": m.lot}
    async def submit(m):  # writes PENDING_APPROVAL only; fails closed when the state store is unavailable
        if state is None:
            raise AssistantError(-32002, "mcp-state-store unavailable; Tier 3 fails closed")
        return (await state.call("workflow_submit", m.model_dump(mode="json"), WorkflowSubmitOut)).model_dump(mode="json")

    return [ToolSpec("hybrid_retrieve", "TIER_1_READ", HybridRetrieveIn, hybrid),
            ToolSpec("rerank", "TIER_1_READ", RerankIn, rerank),
            ToolSpec("graphify_expand", "TIER_1_READ", GraphifyExpandIn, graphify),
            ToolSpec("validate_sku", "TIER_2_COMPUTE", ValidateSkuIn, vsku),
            ToolSpec("validate_lot", "TIER_2_COMPUTE", ValidateLotIn, vlot),
            ToolSpec("workflow_submit", "TIER_3_MUTATE", WorkflowSubmitIn, submit)]
