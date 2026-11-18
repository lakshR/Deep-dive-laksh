import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from reliable_assistant.api import create_app
from reliable_assistant.auth import mint_token
from reliable_assistant.errors import AssistantError
from reliable_assistant.plugins.manager import PluginManager
from reliable_assistant.plugins.specs import Verdict, hookimpl
from reliable_assistant.types import Meta, SkuId

CFG = {"plugins": [
    {"name": "a", "stage": "PRE_EXECUTION", "priority": 20, "fail_closed": True},
    {"name": "b", "stage": "PRE_EXECUTION", "priority": 10, "fail_closed": True},
    {"name": "o", "stage": "ALL_SPANS"}]}


def test_meta_rejects_bad_tenant_and_extra():
    good = dict(tenant_id="acme", user_id_hash="a" * 32, request_id="0" * 26, trace_id="b" * 32,
                deadline_ms=150, idempotency_key="12345678-1234-4234-8234-123456789abc")
    Meta(**good)
    with pytest.raises(ValidationError):
        Meta(**{**good, "tenant_id": "Bad_Tenant"})
    with pytest.raises(ValidationError):
        Meta(**good, extra=1)


def test_plugins_run_in_priority_order_and_deny_short_circuits():
    order = []

    class P:
        def __init__(self, n, deny=False): self.n, self.deny = n, deny
        @hookimpl
        def pre_execute(self, ctx):
            order.append(self.n)
            return Verdict.deny(-32014, "x") if self.deny else Verdict()

    pm = PluginManager(CFG)
    pm.register("a", P("a")); pm.register("b", P("b", deny=True))
    with pytest.raises(AssistantError) as e:
        pm.run_verdicts("PRE_EXECUTION", ctx={})
    assert order == ["b"] and e.value.code == -32014


def test_fail_closed_vs_fail_open():
    class Boom:
        @hookimpl
        def pre_execute(self, ctx): raise RuntimeError("x")
        @hookimpl
        def on_span(self, span): raise RuntimeError("x")

    pm = PluginManager(CFG)
    pm.register("a", Boom()); pm.register("o", Boom())
    with pytest.raises(AssistantError):
        pm.run_verdicts("PRE_EXECUTION", ctx={})
    pm.run_observers("ALL_SPANS", span={})  # must not raise


def test_api_tenant_isolation_and_auth():
    c = TestClient(create_app(with_rag=False, with_state=False))
    body = {"message": "What is the lot recall procedure?"}
    assert c.post("/api/v1/agent/query", json=body).status_code == 401
    h = {"Authorization": f"Bearer {mint_token('acme', 'u1')}"}
    r = c.post("/api/v1/agent/query", json=body, headers=h)
    assert r.status_code == 200 and r.json()["tenant_id"] == "acme" and r.json()["abstained"]
    # asserted tenant must match JWT, in body or header
    assert c.post("/api/v1/agent/query", json={**body, "tenant_id": "other"}, headers=h).status_code == 401
    assert c.post("/api/v1/agent/query", json=body, headers={**h, "X-Tenant-ID": "other"}).status_code == 401
    assert c.post("/api/v1/agent/query", json={"message": ""}, headers=h).status_code == 422


def test_sku_type():
    from pydantic import TypeAdapter
    assert TypeAdapter(SkuId).validate_python("SKU-099")
    with pytest.raises(ValidationError):
        TypeAdapter(SkuId).validate_python("SKU99")
