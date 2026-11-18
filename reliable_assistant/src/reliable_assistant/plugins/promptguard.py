"""Optional second-stage injection classifier (Meta Prompt Guard 2 via the OpenAI-compatible endpoint). Opt-in:
RA_INJECTION_MODEL=meta-llama/llama-prompt-guard-2-86m. Adds ~120 ms per request (breaks the 15 ms KPI-01 budget),
sends the already PII-redacted message to the provider, and is rate-limited — so it FAILS OPEN to the regex layer
(and records that it did). Measured on the held-out set: see evals/results/latest.json."""
import os

import httpx

from ..llm.client import BASE_URL, load_key

THRESHOLD = float(os.environ.get("RA_INJECTION_THRESHOLD", "0.5"))


class PromptGuard:
    def __init__(self, model: str, timeout_s: float = 3.0):
        key = load_key()
        self.model = model
        self.http = httpx.Client(base_url=BASE_URL, timeout=timeout_s, headers={"Authorization": f"Bearer {key}"}) if key else None

    def score(self, text: str) -> float | None:
        if self.http is None:
            return None
        try:
            r = self.http.post("/chat/completions", json={"model": self.model, "max_tokens": 8,
                                                          "messages": [{"role": "user", "content": text[:4000]}]})
            return float(r.json()["choices"][0]["message"]["content"]) if r.status_code == 200 else None
        except Exception:  # noqa: BLE001
            return None


def from_env() -> PromptGuard | None:
    m = os.environ.get("RA_INJECTION_MODEL")
    return PromptGuard(m) if m else None
