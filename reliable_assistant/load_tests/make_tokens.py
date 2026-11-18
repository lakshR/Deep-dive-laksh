"""Mint per-tenant dev JWTs for load tests (design.md Appendix B #11: the load test must mint per-tenant JWTs; the
tenant in body/header is only an assertion). Usage: python load_tests/make_tokens.py [n_extra_tenants]
NOTE: the blueprint's k6 script uses tenant ids like 'tenant_7'; underscores are invalid under TenantId, so every
request would 401. We use 'tenant-7'. Extra tenants have EMPTY vaults (answers abstain) — they exercise isolation and
the gateway, not retrieval."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from reliable_assistant.auth import mint_token  # noqa: E402

extra = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tenants = ["acme", "globex"] + [f"tenant-{i}" for i in range(extra)]
out = [{"tenant": t, "token": mint_token(t, f"load-user-{i}")} for i in range(25) for t in tenants]
(ROOT / "load_tests" / "tokens.json").write_text(json.dumps(out))
(ROOT / "load_tests" / "tokens.csv").write_text("".join(f"{o['tenant']},{o['token']}\n" for o in out))   # for JMeter CSV Data Set
print(f"wrote {len(out)} tokens for {len(tenants)} tenants -> load_tests/tokens.json (gitignore it)")
