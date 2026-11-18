import json
import os

import pytest
from fastapi.testclient import TestClient

from reliable_assistant.api import create_app
from reliable_assistant.auth import mint_token
from reliable_assistant.llm.client import load_key, make_client
from reliable_assistant.llm.ground import (INSUFFICIENT, build_messages, enforce_citations, generate_grounded)

CH = [{"chunk_id": "acme:lot-recall-sop:0", "text": "Quarantine remaining units within 4 hours. Contact qa.lead@acme.com."},
      {"chunk_id": "acme:sku-labeling-rules:0", "text": "Labels go on the top-right face of the carton."}]


class FakeLlm:
    """Scripted: generator returns `gen`, verifier returns `ver`."""
    def __init__(self, gen, ver='{"supported": [true]}'):
        self.gen, self.ver, self.calls = gen, ver, []

    async def chat(self, model, messages, max_tokens=400, json_mode=False):
        self.calls.append((model, messages))
        return (self.ver if json_mode else self.gen), 5.0


def run(llm):
    import asyncio
    return asyncio.run(generate_grounded(llm, "gen", "ver", "How fast must lots be quarantined?", ["c"], CH))


def test_enforce_citations_drops_uncited_and_invalid():
    ids = {"acme:lot-recall-sop:0"}
    got = enforce_citations("Quarantine within 4 hours [acme:lot-recall-sop:0]. Invented fact. Bad ref [acme:x:9].", ids)
    assert got == [("Quarantine within 4 hours.", ["acme:lot-recall-sop:0"])]   # uncited + invalid-marker sentences removed


def test_marker_after_period_and_unicode_hyphen():
    got = enforce_citations("LOT‑88392 must be held within 4 hours. [acme:lot-recall-sop:0]", {"acme:lot-recall-sop:0"})
    assert got == [("LOT-88392 must be held within 4 hours.", ["acme:lot-recall-sop:0"])]


def test_ok_path():
    r = run(FakeLlm("Units must be quarantined within 4 hours [acme:lot-recall-sop:0]."))
    assert r["status"] == "ok" and r["faithfulness"] == 1.0 and r["citations"] == ["acme:lot-recall-sop:0"]
    assert "[" not in r["answer"]


def test_insufficient_evidence():
    assert run(FakeLlm(INSUFFICIENT))["status"] == "insufficient"


def test_all_uncited_is_ungrounded():
    assert run(FakeLlm("Paris is the capital of France."))["status"] == "ungrounded"


def test_verifier_veto():
    r = run(FakeLlm("Units must be quarantined within 4 hours [acme:lot-recall-sop:0].", '{"supported": [false]}'))
    assert r["status"] == "ungrounded" and r["faithfulness"] == 0.0 and r["answer"] == ""


@pytest.mark.parametrize("bad", ["not json", '{"supported": [true, true]}', '{"x": 1}', '{"supported": ["yes"]}'])
def test_verifier_garbage_fails_closed(bad):
    assert run(FakeLlm("Units must be quarantined within 4 hours [acme:lot-recall-sop:0].", bad))["status"] == "ungrounded"


def test_partial_support_below_threshold_is_vetoed():
    gen = "Quarantine within 4 hours [acme:lot-recall-sop:0]. Labels go on the carton [acme:sku-labeling-rules:0]."
    assert run(FakeLlm(gen, '{"supported": [true, false]}'))["status"] == "ungrounded"   # 0.5 < 0.92


def test_prompt_redacts_pii_and_marks_evidence_as_data():
    msgs = build_messages("q", ["c"], CH)
    blob = json.dumps(msgs)
    assert "qa.lead@acme.com" not in blob and "[EMAIL]" in blob
    assert "<evidence id=" in msgs[1]["content"] and "untrusted data" in msgs[0]["content"]
    assert msgs[1]["content"].startswith("GOAL:")           # GoalAnchorBlock first


# ---- live (real key; skipped when none is available) ----
pytestmark_live = pytest.mark.skipif(load_key() is None, reason="no GROQ_API_KEY / RA_ENV_FILE")


@pytest.fixture
def skip_on_rate_limit():
    from reliable_assistant.errors import AssistantError
    try:
        yield
    except AssistantError as e:
        if e.code == -32003:
            pytest.skip("LLM provider rate limited")
        raise


@pytestmark_live
def test_live_end_to_end(skip_on_rate_limit):
    with TestClient(create_app(with_rag=True, with_state=False, llm=make_client())) as c:
        h = {"Authorization": f"Bearer {mint_token('acme', 'u1')}"}
        j = c.post("/api/v1/agent/query", json={"message": "How fast must LOT-88392 be quarantined?"}, headers=h).json()
        assert not j["abstained"] and j["faithfulness"] >= 0.92 and "4 hours" in j["answer"]
        assert j["citations"][0]["chunk_id"].startswith("acme:lot-recall-sop")
        assert "[" not in j["answer"] and "LOT-88392" in j["answer"]
        n = c.post("/api/v1/agent/query", json={"message": "What is the capital of France?"}, headers=h).json()
        assert n["abstained"]
        g = c.post("/api/v1/agent/query", json={"message": "How fast must recalled lots be quarantined?"},
                   headers={"Authorization": f"Bearer {mint_token('globex', 'u1')}"}).json()
        assert "24 hours" in g["answer"] and "acme" not in json.dumps(g)
        # Globex's document never mentions LOT-88392: the model must not borrow Acme's answer
        x = c.post("/api/v1/agent/query", json={"message": "How fast must LOT-88392 be quarantined?"},
                   headers={"Authorization": f"Bearer {mint_token('globex', 'u1')}"}).json()
        assert "4 hours" not in x["answer"].replace("24 hours", "")


@pytestmark_live
def test_live_verifier_catches_wrong_claim(skip_on_rate_limit):
    import asyncio
    from reliable_assistant.llm.client import VERIFY_MODEL
    from reliable_assistant.llm.ground import verify
    by_id = {"acme:lot-recall-sop:0": {"text": "Warehouse staff must quarantine all remaining units within 4 hours."}}
    async def go():
        llm = make_client()
        try:
            good = await verify(llm, VERIFY_MODEL, [("Units must be quarantined within 4 hours.", ["acme:lot-recall-sop:0"])], by_id)
            bad = await verify(llm, VERIFY_MODEL, [("Units must be quarantined within 24 hours.", ["acme:lot-recall-sop:0"])], by_id)
        finally:
            await llm.aclose()
        return good, bad
    good, bad = asyncio.run(go())
    assert good == 1.0 and bad == 0.0
