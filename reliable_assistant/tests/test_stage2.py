from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reliable_assistant.api import create_app
from reliable_assistant.auth import mint_token
from reliable_assistant.retrieval.index import rrf, tokenize

ROOT = Path(__file__).resolve().parents[1]


def test_rrf_matches_formula_and_is_unweighted():
    out = rrf([["a", "b"], ["b", "c"]], k=60)
    assert out["a"] == pytest.approx(1 / 61)
    assert out["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert max(out, key=out.get) == "b"


def test_tokenizer_stems():
    assert tokenize("Quarantined lots") == tokenize("quarantine lot")


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(with_rag=True, with_state=False, llm=None)) as c:  # spawns the real MCP stdio server
        yield c


def hdr(tenant):
    return {"Authorization": f"Bearer {mint_token(tenant, 'u1')}"}


def test_grounded_answer_with_citation(client):
    r = client.post("/api/v1/agent/query", json={"message": "How fast must LOT-88392 be quarantined?"},
                    headers=hdr("acme"))
    j = r.json()
    assert r.status_code == 200 and not j["abstained"]
    assert j["citations"][0]["chunk_id"].startswith("acme:lot-recall-sop")
    assert "within 4 hours" in j["answer"]


def test_off_topic_abstains(client):
    j = client.post("/api/v1/agent/query", json={"message": "What is the capital of France?"},
                    headers=hdr("acme")).json()
    assert j["abstained"] and j["citations"] == []


def test_tenant_isolation(client):
    a = client.post("/api/v1/agent/query", json={"message": "How fast must LOT-88392 be quarantined?"},
                    headers=hdr("acme")).json()
    g = client.post("/api/v1/agent/query", json={"message": "How fast must LOT-88392 be quarantined?"},
                    headers=hdr("globex")).json()
    assert "within 4 hours" in a["answer"] and "Globex" not in a["answer"]
    assert "within 24 hours" in g["answer"] and "within 4 hours" not in g["answer"]
    assert "id:" not in a["answer"] and "---" not in a["answer"]  # frontmatter stripped
    assert all(c["chunk_id"].startswith("globex:") for c in g["citations"])
    # a tenant with no vault gets nothing
    n = client.post("/api/v1/agent/query", json={"message": "quarantine lot"}, headers=hdr("nobody")).json()
    assert n["abstained"]
