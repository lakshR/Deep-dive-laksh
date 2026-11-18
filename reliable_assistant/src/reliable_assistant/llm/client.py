"""Minimal async client for OpenAI-compatible chat endpoints (Groq by default).
The API key is read from the environment at runtime — or from the file named by RA_ENV_FILE — and is never
written into this project or logged."""
import os
import time
from pathlib import Path

import httpx

from ..errors import AssistantError

BASE_URL = os.environ.get("RA_LLM_BASE_URL", "https://api.groq.com/openai/v1")
GEN_MODEL = os.environ.get("RA_LLM_MODEL", "openai/gpt-oss-20b")
VERIFY_MODEL = os.environ.get("RA_VERIFY_MODEL", "qwen/qwen3.8-27b")  # independent model for the Verification Subagent


def load_key(var: str = "GROQ_API_KEY") -> str | None:
    if os.environ.get(var):
        return os.environ[var]
    path = os.environ.get("RA_ENV_FILE")
    if path and Path(path).expanduser().is_file():
        for line in Path(path).expanduser().read_text().splitlines():
            if line.startswith(var + "="):
                return line.split("=", 1)[1].strip().strip("'\"") or None
    return None


class LlmClient:
    def __init__(self, api_key: str, timeout_s: float = 20.0):
        self._http = httpx.AsyncClient(base_url=BASE_URL, timeout=timeout_s,
                                       headers={"Authorization": f"Bearer {api_key}"})

    async def aclose(self):
        await self._http.aclose()

    async def chat(self, model: str, messages: list[dict], max_tokens: int = 400, json_mode: bool = False) -> tuple[str, float]:
        body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
                "reasoning_effort": "low", "include_reasoning": False}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            r = await self._http.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise AssistantError(-32002, f"llm unreachable: {exc.__class__.__name__}") from exc
        ms = (time.perf_counter() - t0) * 1e3
        if r.status_code == 429:
            raise AssistantError(-32003, "llm rate limited")
        if r.status_code >= 400:
            raise AssistantError(-32002, f"llm error HTTP {r.status_code}: {r.text[:200]}")
        return (r.json()["choices"][0]["message"].get("content") or "").strip(), ms


def make_client() -> LlmClient | None:
    key = load_key()
    return LlmClient(key) if key else None
