"""Bounded ReAct state machine (ADR-4): INIT -> TRIAGE -> REASONING -> TOOL_CALL -> OBSERVATION -> VERIFY -> DONE/HALTED.
Caps live here, outside any prompt: max_iterations=5, per-step timeout, total loop budget."""
import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

from ..errors import AssistantError
from ..types import GoalAnchorBlock, TriageOut
from .compress import compress_observation
from .harness import Grant, ToolExecutionHarness


@dataclass(frozen=True)
class LoopBudget:
    max_iterations: int = 5
    step_timeout_ms: int = 120     # design.md ADR-4
    total_ms: int = 450


# Candidates sent to the cross-encoder. Doc: 40. On CPU, rerank time is ~linear in this (evals/rerank_pool.py:
# 5 -> 219 ms p50, 40 -> 1768 ms). The dev launcher sets 10; the default stays at the design value.
RERANK_POOL = int(os.environ.get("RA_RERANK_POOL", "40"))

# CPU-only dev host cannot meet 120/450 ms (no GPU, FP32). Select with RA_LOOP_PROFILE=dev|spec.
DEV_BUDGET = LoopBudget(5, 5000, 15000)


class State(str, Enum):
    INIT = "INIT"; TRIAGE = "TRIAGE"; REASONING = "REASONING"; TOOL_CALL = "TOOL_CALL"
    OBSERVATION = "OBSERVATION"; VERIFY = "VERIFY"; DONE = "DONE"
    HALTED_BREAKER = "HALTED_BREAKER"; HALTED_BUDGET = "HALTED_BUDGET"; HALTED_HITL = "HALTED_HITL"
    HALTED_ERROR = "HALTED_ERROR"


RETRIEVAL = Grant("retrieval", frozenset({"hybrid_retrieve", "rerank", "graphify_expand"}), "TIER_1_READ")
EXECUTION = Grant("execution", frozenset({"validate_sku", "validate_lot", "workflow_submit"}), "TIER_3_MUTATE")


@dataclass
class Action:
    tool: str
    args: dict
    grant: Grant
    optional: bool = False     # failure is non-fatal (e.g. graph expansion): degrade, don't halt


@dataclass
class Final:
    pass


@dataclass
class LoopRun:
    state: State = State.INIT
    steps: list[dict] = field(default_factory=list)       # one entry per tool call (KPI-17)
    results: dict[str, dict] = field(default_factory=dict)
    observations: list[str] = field(default_factory=list)
    halt_code: int | None = None
    hitl_ticket_id: str | None = None
    trace: list[State] = field(default_factory=list)
    iterations: int = 0                                   # ReAct steps taken (KPI-17: mean ≤ 2.2)

    def to(self, s: State):
        self.state = s
        self.trace.append(s)


class Policy(Protocol):
    def next(self, run: LoopRun, anchor: GoalAnchorBlock, triage: TriageOut) -> Action | list[Action] | Final: ...


class DeterministicPolicy:
    """ADR-4(a): for HYBRID_RAG the orchestrator, not an LLM, performs the first retrieval.
    An LLM-backed policy can replace this behind the same interface."""

    def next(self, run, anchor, triage):
        q = anchor.original_intent
        if triage.route == "HYBRID_RAG":
            if "hybrid_retrieve" not in run.results:
                return Action("hybrid_retrieve", {"query": q}, RETRIEVAL)
            chunks = run.results["hybrid_retrieve"]["chunks"]
            if chunks and "rerank" not in run.results:
                seeds = [c["chunk_id"] for c in chunks[:3]]        # graph expansion runs concurrently with rerank
                return [Action("rerank", {"query": q, "candidates": chunks[:RERANK_POOL]}, RETRIEVAL),
                        Action("graphify_expand", {"seeds": seeds, "hops": 2, "token_budget": 320,
                                                   "edge_types": ["wikilink"]}, RETRIEVAL, optional=True)]
        elif triage.route == "TRANSACTION_EXEC" and "workflow_submit" not in run.results:
            return Action("workflow_submit", {"workflow": triage.intent,
                                              "payload": {"entities": anchor.normalized_entities}}, EXECUTION)
        return Final()


class ReActLoop:
    def __init__(self, harness: ToolExecutionHarness, budget: LoopBudget, policy: Policy | None = None):
        self.harness, self.budget, self.policy = harness, budget, policy or DeterministicPolicy()

    async def run(self, ctx: dict, anchor: GoalAnchorBlock, triage: TriageOut) -> LoopRun:
        run = LoopRun()
        run.to(State.TRIAGE)
        ctx = {**ctx, "breaker_counts": {}}
        t0 = time.perf_counter()
        for _ in range(self.budget.max_iterations):
            run.to(State.REASONING)
            action = self.policy.next(run, anchor, triage)
            if isinstance(action, Final):
                run.to(State.VERIFY); run.to(State.DONE)
                return run
            actions = action if isinstance(action, list) else [action]
            remaining_ms = self.budget.total_ms - (time.perf_counter() - t0) * 1e3
            if remaining_ms <= 0:
                return self._halt(run, State.HALTED_BUDGET, -32031)
            run.to(State.TOOL_CALL)                      # one loop iteration == one step, even if calls run in parallel
            timeout = min(self.budget.step_timeout_ms, remaining_ms) / 1e3
            started, started_ns = time.perf_counter(), time.time_ns()

            async def one(a: Action):
                args = {**a.args, "meta": {
                    "tenant_id": ctx["tenant_id"], "user_id_hash": ctx["user_id_hash"],
                    "request_id": ctx["request_id"], "trace_id": ctx["trace_id"],
                    "deadline_ms": int(max(10, min(2000, remaining_ms))), "idempotency_key": str(uuid.uuid4())}}
                return await self.harness.call(a.tool, args, ctx, a.grant)

            gathered = await asyncio.gather(*[asyncio.wait_for(one(a), timeout) for a in actions], return_exceptions=True)
            ms = (time.perf_counter() - started) * 1e3
            fatal = None
            for a, res in zip(actions, gathered):
                if isinstance(res, BaseException):
                    code = res.code if isinstance(res, AssistantError) else -32001 if isinstance(res, asyncio.TimeoutError) else -32603
                    run.steps.append({"tool": a.tool, "ms": ms, "error": code, "start_ns": started_ns})
                    if a.optional and code not in (-32032, -32015):
                        continue
                    fatal = fatal or (a, res, code)
                else:
                    run.steps.append({"tool": a.tool, "ms": ms, "start_ns": started_ns})
                    run.results[a.tool] = res
                    run.observations.append(compress_observation(str(res), anchor.original_intent))  # ≤180 tok
            if fatal:
                a, exc, code = fatal
                if isinstance(exc, asyncio.TimeoutError):
                    return self._halt(run, State.HALTED_BUDGET, -32031)
                if code == -32032:
                    return self._halt(run, State.HALTED_BREAKER, code)
                if code == -32015:
                    run.hitl_ticket_id = exc.data.get("hitl_ticket_id")
                    return self._halt(run, State.HALTED_HITL, code)
                return self._halt(run, State.HALTED_ERROR, code)
            run.to(State.OBSERVATION)
            run.iterations += 1
        return self._halt(run, State.HALTED_BUDGET, -32031)  # iteration cap hit without Final

    @staticmethod
    def _halt(run: LoopRun, state: State, code: int) -> LoopRun:
        run.to(state)
        run.halt_code = code
        return run
