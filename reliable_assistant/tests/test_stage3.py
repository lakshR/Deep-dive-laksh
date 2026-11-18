import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from reliable_assistant.agent.entities import extract_entities, repair_str
from reliable_assistant.agent.harness import Grant, ToolExecutionHarness, ToolSpec
from reliable_assistant.agent.loop import (DEV_BUDGET, EXECUTION, RETRIEVAL, Action, Final, LoopBudget, ReActLoop,
                                           State)
from reliable_assistant.agent.triage import triage
from reliable_assistant.api import build_plugins, create_app
from reliable_assistant.auth import mint_token
from reliable_assistant.errors import AssistantError
from reliable_assistant.types import GoalAnchorBlock, ValidateSkuIn, WorkflowSubmitIn, ValidateLotIn


def meta(tenant="acme"):
    return dict(tenant_id=tenant, user_id_hash="a" * 32, request_id="0" * 26, trace_id="b" * 32, deadline_ms=500,
                idempotency_key=str(uuid.uuid4()))


CTX = {"tenant_id": "acme", "user_id_hash": "a" * 32, "request_id": "0" * 26, "trace_id": "b" * 32}


def make_harness(handlers=None):
    calls = []

    async def vsku(m): calls.append("sku"); return {"valid": True, "sku": m.sku}
    async def vlot(m): return {"valid": True, "lot": m.lot}
    async def submit(m): calls.append("pending_written"); return {"status": "PENDING_APPROVAL", "hitl_ticket_id": "HITL-TEST"}
    specs = [ToolSpec("validate_sku", "TIER_2_COMPUTE", ValidateSkuIn, vsku),
             ToolSpec("validate_lot", "TIER_2_COMPUTE", ValidateLotIn, vlot),
             ToolSpec("workflow_submit", "TIER_3_MUTATE", WorkflowSubmitIn, submit)]
    return ToolExecutionHarness(specs, build_plugins()), calls


# ---- auto-repair ----
@pytest.mark.parametrize("raw,fixed", [("SKU99", "SKU-099"), ("sku 99", "SKU-099"), ("lot 44", "LOT-00044"),
                                       ("LOT-44", "LOT-00044"), ("SKU-1234", "SKU-1234")])
def test_repair_whole_values(raw, fixed):
    assert repair_str(raw) == fixed


def test_prose_is_not_rewritten():
    assert repair_str("we shipped a lot 5 times") == "we shipped a lot 5 times"
    assert repair_str("check SKU99 now") == "check SKU-099 now"
    assert extract_entities("verify lot 44 and sku99") == {"lot": "LOT-00044", "sku": "SKU-099"}


async def test_harness_repairs_then_validates():
    h, calls = make_harness()
    r = await h.call("validate_sku", {"meta": meta(), "sku": "SKU99"}, dict(CTX), EXECUTION)
    assert r == {"valid": True, "sku": "SKU-099"}
    assert h.metrics["first_pass_fail"] == 1 and h.metrics["repair_ok"] == 1 and h.audit[0]["original"]["sku"] == "SKU99"


async def test_harness_rejects_unrepairable_and_unknown():
    h, _ = make_harness()
    with pytest.raises(AssistantError) as e:
        await h.call("validate_sku", {"meta": meta(), "sku": "banana"}, dict(CTX), EXECUTION)
    assert e.value.code == -32602
    with pytest.raises(AssistantError) as e:
        await h.call("nope", {}, dict(CTX), EXECUTION)
    assert e.value.code == -32601


async def test_rbac_retrieval_cannot_use_execution_tools():
    h, _ = make_harness()
    with pytest.raises(AssistantError) as e:
        await h.call("validate_sku", {"meta": meta(), "sku": "SKU-099"}, dict(CTX), RETRIEVAL)
    assert e.value.code == -32601
    low = Grant("lowtier", frozenset({"validate_sku"}), "TIER_1_READ")
    with pytest.raises(AssistantError) as e:
        await h.call("validate_sku", {"meta": meta(), "sku": "SKU-099"}, dict(CTX), low)
    assert e.value.code == -32012


async def test_tenant_mismatch_and_non_v4_key():
    h, _ = make_harness()
    with pytest.raises(AssistantError) as e:
        await h.call("validate_sku", {"meta": meta("globex"), "sku": "SKU-099"}, dict(CTX), EXECUTION)
    assert e.value.code == -32011
    m = meta(); m["idempotency_key"] = str(uuid.uuid1())
    with pytest.raises(AssistantError) as e:
        await h.call("validate_sku", {"meta": m, "sku": "SKU-099"}, dict(CTX), EXECUTION)
    assert e.value.code == -32602


async def test_idempotency_replay_does_not_reexecute():
    h, calls = make_harness()
    args = {"meta": meta(), "sku": "SKU-099"}
    await h.call("validate_sku", args, dict(CTX), EXECUTION)
    await h.call("validate_sku", args, dict(CTX), EXECUTION)
    assert calls == ["sku"] and h.metrics["idempotent_replays"] == 1


async def test_tier3_never_executes_and_returns_ticket():
    h, calls = make_harness()
    with pytest.raises(AssistantError) as e:
        await h.call("workflow_submit", {"meta": meta(), "workflow": "refund_request", "payload": {}},
                     {**CTX, "breaker_counts": {}}, EXECUTION)
    assert e.value.code == -32015 and e.value.data["hitl_ticket_id"] == "HITL-TEST"
    assert calls == ["pending_written"]   # only the PENDING_APPROVAL write ran; no action is ever executed


async def test_tier3_blocked_even_if_hitl_plugin_disabled():
    h, calls = make_harness()
    h.plugins.set_enabled("hitl", False)
    with pytest.raises(AssistantError) as e:
        await h.call("workflow_submit", {"meta": meta(), "workflow": "refund_request", "payload": {}},
                     {**CTX, "breaker_counts": {}}, EXECUTION)
    assert e.value.code == -32015 and calls == ["pending_written"]


# ---- loop: chaos suite (KPI-16 = 100 %) ----
class Repeater:
    def next(self, run, anchor, triage):
        return Action("validate_sku", {"sku": "SKU-099"}, EXECUTION)   # identical forever


ANCHOR = GoalAnchorBlock(original_intent="x", normalized_entities={}, policy_constraints=[])
TRI = triage("What is the lot recall procedure?")


async def test_infinite_loop_is_stopped_by_breaker():
    h, calls = make_harness()
    run = await ReActLoop(h, DEV_BUDGET, Repeater()).run(dict(CTX), ANCHOR, TRI)
    assert run.state == State.HALTED_BREAKER and run.halt_code == -32032
    assert len(calls) == 2   # max_identical_calls=2 executed, 3rd denied


class Varying:
    n = 0
    def next(self, run, anchor, triage):
        Varying.n += 1
        return Action("validate_lot", {"lot": f"LOT-{Varying.n:05d}"}, EXECUTION)


async def test_iteration_cap_when_args_keep_changing():
    h, _ = make_harness()
    Varying.n = 0
    run = await ReActLoop(h, DEV_BUDGET, Varying()).run(dict(CTX), ANCHOR, TRI)
    assert run.state == State.HALTED_BUDGET and run.halt_code == -32031 and len(run.steps) == 5


async def test_step_timeout_halts():
    async def slow(m): await asyncio.sleep(1); return {}
    h = ToolExecutionHarness([ToolSpec("validate_sku", "TIER_2_COMPUTE", ValidateSkuIn, slow)], build_plugins())
    run = await ReActLoop(h, LoopBudget(5, 120, 450), Repeater()).run(dict(CTX), ANCHOR, TRI)
    assert run.state == State.HALTED_BUDGET and run.halt_code == -32031


# ---- triage ----
@pytest.mark.parametrize("msg,route,tier", [
    ("What is the refund policy?", "HYBRID_RAG", "TIER_1_READ"),
    ("How fast must LOT-88392 be quarantined?", "HYBRID_RAG", "TIER_1_READ"),
    ("Please approve the refund for order 123", "TRANSACTION_EXEC", "TIER_3_MUTATE"),
    ("Refund customer 77 now", "TRANSACTION_EXEC", "TIER_3_MUTATE"),
    ("I need to adjust the dose for patient 5", "TRANSACTION_EXEC", "TIER_3_MUTATE"),
    ("hello", "DIRECT_ANSWER", "TIER_1_READ"),
    ("tell me a joke", "REJECT_OOS", "TIER_1_READ"),
])
def test_triage_routes(msg, route, tier):
    t = triage(msg)
    assert (t.route, t.risk_tier) == (route, tier)


# ---- API end to end ----
@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(with_rag=True, with_state=False, llm=None)) as c:
        yield c


def hdr(t="acme"): return {"Authorization": f"Bearer {mint_token(t, 'u1')}"}


def test_api_tier3_fails_closed_without_state_store(client):
    r = client.post("/api/v1/agent/query", json={"message": "Please approve the refund for order 123"}, headers=hdr())
    assert r.status_code == 503 and r.json()["code"] == -32002   # no ticket possible -> request not accepted


def test_api_domain_disclaimer_and_oos_and_greeting(client):
    j = client.post("/api/v1/agent/query", json={"message": "What is the refund policy for customers?"}, headers=hdr()).json()
    assert j["answer"].startswith("Informational only") and not j["abstained"]
    assert client.post("/api/v1/agent/query", json={"message": "tell me a joke"}, headers=hdr()).json()["halt_reason"] == "REJECT_OOS"
    assert not client.post("/api/v1/agent/query", json={"message": "hello"}, headers=hdr()).json()["abstained"]
