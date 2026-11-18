"""Shared constrained types — design.md §1.2.1 (verbatim contracts)."""
from typing import Annotated, Literal
from uuid import UUID

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints as SC, field_validator

TenantId = Annotated[str, SC(pattern=r"^[a-z0-9][a-z0-9-]{2,31}$")]
UserHash = Annotated[str, SC(pattern=r"^[a-f0-9]{32}$")]
SessionId = Annotated[str, SC(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")]
RequestId = Annotated[str, SC(pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$")]
ChunkId = Annotated[str, SC(pattern=r"^[a-z0-9-]{3,32}:[A-Za-z0-9_.-]{1,64}:\d{1,6}$")]
ChunkHash = Annotated[str, SC(pattern=r"^[a-f0-9]{64}$")]
def _no_traversal(v: str) -> str:
    if ".." in v or v.startswith(("/", "~")) or "\\" in v:
        raise ValueError("path traversal not allowed in note slug")
    return v


NoteSlug = Annotated[str, SC(pattern=r"^[\w\-./ ]{1,200}$"), AfterValidator(_no_traversal)]
QueryText = Annotated[str, SC(min_length=1, max_length=4000, strip_whitespace=True)]
SkuId = Annotated[str, SC(pattern=r"^SKU-\d{3,6}$")]
LotId = Annotated[str, SC(pattern=r"^LOT-\d{5}$")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Meta(Strict):
    tenant_id: TenantId
    user_id_hash: UserHash
    request_id: RequestId
    trace_id: Annotated[str, SC(pattern=r"^[a-f0-9]{32}$")]
    deadline_ms: Annotated[int, Field(ge=10, le=2000)]
    idempotency_key: UUID


class Citation(Strict):
    chunk_id: ChunkId
    uri: Annotated[str, SC(pattern=r"^rag://chunks/[A-Za-z0-9:_.-]+$")]
    quote_span: tuple[int, int]


class TriageOut(Strict):
    intent: str
    domain: Literal["general", "clinical", "financial", "legal", "inventory", "other"]
    risk_tier: Literal["TIER_1_READ", "TIER_2_COMPUTE", "TIER_3_MUTATE"]
    complexity: Literal["low", "medium", "high"]
    route: Literal["DIRECT_ANSWER", "HYBRID_RAG", "TRANSACTION_EXEC", "REJECT_OOS"]


class GoalAnchorBlock(Strict):
    original_intent: QueryText
    normalized_entities: dict[str, str]
    policy_constraints: Annotated[list[str], Field(max_length=10)]


class SessionFact(Strict):
    fact: Annotated[str, Field(max_length=300)]
    source_chunk_id: ChunkId
    verified: Literal[True] = True


class SessionState(Strict):
    session_id: SessionId
    turn_index: Annotated[int, Field(ge=0, le=500)]
    goal_anchor: GoalAnchorBlock
    last_turns: Annotated[list[dict], Field(max_length=3)]
    schema_version: Literal[1] = 1


class AgentQueryIn(Strict):
    """Public API body (§1.2.8). tenant_id is only a consistency assertion vs the JWT."""
    session_id: SessionId | None = None
    message: QueryText
    tenant_id: TenantId | None = None


class AgentQueryOut(Strict):
    answer: str
    citations: list[Citation]
    route: str
    tenant_id: TenantId
    faithfulness: float | None
    abstained: bool
    trace_id: Annotated[str, SC(pattern=r"^[a-f0-9]{32}$")]
    hitl_ticket_id: str | None = None
    halt_reason: str | None = None
    session_id: SessionId | None = None


# ---- Stage 2: retrieval contracts (§1.2.2 / §1.2.3) ----
class HybridRetrieveIn(Strict):
    meta: Meta
    query: QueryText
    sparse_top_k: Annotated[int, Field(ge=1, le=100)] = 50
    dense_top_k: Annotated[int, Field(ge=1, le=100)] = 50
    bm25_k1: Annotated[float, Field(ge=0.5, le=2.5)] = 1.25
    bm25_b: Annotated[float, Field(ge=0, le=1)] = 0.75
    rrf_k: Annotated[int, Field(ge=1, le=200)] = 60
    pool: Annotated[int, Field(ge=1, le=40)] = 40   # doc says le=100 but HybridRetrieveOut caps chunks at 40 -> >40 crashed; capped here
    filters: dict[Annotated[str, SC(pattern=r"^[a-z_]{1,32}$")], str | int | bool] = {}


class RetrievedChunk(Strict):
    chunk_id: ChunkId
    chunk_hash: ChunkHash
    text: Annotated[str, Field(max_length=3000)]
    bm25_score: float | None
    cosine_score: float | None
    rrf_score: float
    source_uri: str
    acl_fingerprint: Annotated[str, SC(pattern=r"^[a-f0-9]{16}$")]


class HybridRetrieveOut(Strict):
    chunks: Annotated[list[RetrievedChunk], Field(max_length=40)]
    bm25_ms: float
    dense_ms: float
    rrf_ms: float
    degraded: Literal["none", "dense_only", "sparse_only"] = "none"


class RerankIn(Strict):
    meta: Meta
    query: QueryText
    candidates: Annotated[list[RetrievedChunk], Field(min_length=1, max_length=40)]
    rerank_k: Annotated[int, Field(ge=1, le=8)] = 4
    threshold: Annotated[float, Field(ge=0, le=1)] = 0.42


class RerankOut(Strict):
    ranked: list[tuple[ChunkId, float]]
    top_score: float
    abstain: bool
    precision: Literal["fp16", "fp32"]
    degraded: bool = False
    all_scores: list[tuple[ChunkId, float]] = []   # every candidate's score (RAG Lab); `ranked` is the thresholded subset


# ---- Stage 3: execution tools (§1.2.5 / §1.2.6) ----
class WorkflowSubmitIn(Strict):  # TIER_3_MUTATE; never executes directly
    meta: Meta
    workflow: Annotated[str, SC(pattern=r"^[a-z_]{3,48}$")]
    payload: dict


class WorkflowSubmitOut(Strict):
    status: Literal["PENDING_APPROVAL"]
    hitl_ticket_id: str


class ValidateSkuIn(Strict):  # TIER_2_COMPUTE
    meta: Meta
    sku: SkuId


class ValidateLotIn(Strict):  # TIER_2_COMPUTE
    meta: Meta
    lot: LotId


# ---- Stage 4: state-store tools (§1.2.5) ----
class StateGetIn(Strict):
    meta: Meta
    session_id: SessionId


class StateGetOut(Strict):
    state: SessionState | None
    version: int
    facts: list[dict]


class StatePutIn(Strict):
    meta: Meta
    state: SessionState
    expected_version: Annotated[int, Field(ge=0)]       # CAS; mismatch -> -32021
    new_facts: list[SessionFact] = []


class StatePutOut(Strict):
    version: int


class AuditAppendIn(Strict):
    meta: Meta
    query_redacted: str
    evidence_ids: list[ChunkId]
    tool_calls: list[dict]
    answer_redacted: str
    faithfulness: float | None
    plugin_decisions: list[dict]


class AuditAppendOut(Strict):
    seq: int
    record_hash: str    # SHA-256(prev_hash + canonical payload), tamper-evident


class AuditVerifyOut(Strict):
    ok: bool
    first_bad_seq: int | None
    records: int


class HitlDecideIn(Strict):
    meta: Meta
    ticket_id: str
    approve: bool


class HitlDecideOut(Strict):
    status: Literal["APPROVED", "REJECTED"]


# ---- Stage 5: graphify_expand (§1.2.4) ----
class GraphifyExpandIn(Strict):
    meta: Meta
    seeds: Annotated[list[NoteSlug | ChunkId], Field(min_length=1, max_length=8)]
    hops: Annotated[int, Field(ge=1, le=2)] = 2
    token_budget: Annotated[int, Field(ge=200, le=2000)] = 320
    edge_types: list[Literal["wikilink", "import", "call", "backlink"]] = ["wikilink", "import"]


class GraphifyExpandOut(Strict):
    nodes: list[dict]            # {id, chunk_id, summary, edge_from, hop, risk_tier, version}
    tokens_used: int
    tokens_saved_vs_raw: int


class AuditRecentIn(Strict):
    meta: Meta
    limit: Annotated[int, Field(ge=1, le=500)] = 100


class AuditRecentOut(Strict):
    records: list[dict]
