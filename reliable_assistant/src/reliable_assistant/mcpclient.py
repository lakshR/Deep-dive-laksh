"""Persistent MCP client sessions (stdio for local workers, Streamable HTTP for the state store)."""
import json
import re
import sys
import time
from collections import deque
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from .errors import CODES, AssistantError


class McpClient:
    def __init__(self, name: str):
        self.name, self.stack, self.session = name, AsyncExitStack(), None
        self.log: deque = deque(maxlen=200)   # JSON-RPC payload inspector (Streamlit Tab 3)

    async def start_stdio(self, script: str):
        read, write = await self.stack.enter_async_context(
            stdio_client(StdioServerParameters(command=sys.executable, args=[script])))
        await self._init(read, write)

    async def start_http(self, url: str):
        streams = await self.stack.enter_async_context(streamable_http_client(url, max_sse_event_size=16 * 1024 * 1024))   # default 1 MiB; MCP sends results twice
        await self._init(streams[0], streams[1])

    async def _init(self, read, write):
        self.session = await self.stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()

    async def stop(self):
        await self.stack.aclose()

    async def call_raw(self, tool: str, arguments: dict):
        t0, entry = time.perf_counter(), {"ts": time.time(), "server": self.name, "tool": tool,
                                          "request": json.dumps(arguments, default=str)[:1500]}
        try:
            out = await self._call_raw(tool, arguments)
            entry["response"] = json.dumps(out, default=str)[:1500]
            return out
        except Exception as exc:
            entry["error"] = str(exc)[:300]
            raise
        finally:
            entry["ms"] = round((time.perf_counter() - t0) * 1e3, 2)
            self.log.append(entry)

    async def _call_raw(self, tool: str, arguments: dict):
        try:
            res = await self.session.call_tool(tool, arguments)
        except Exception as exc:  # transport failure
            raise AssistantError(-32002, f"{self.name}.{tool} unavailable: {exc.__class__.__name__}") from exc
        if res.is_error:
            text = res.content[0].text if res.content else ""
            m = re.search(r"\[(-\d{5})\]\s*(.*)", text)
            if m and int(m.group(1)) in CODES:   # preserve the server-side AssistantError code
                raise AssistantError(int(m.group(1)), m.group(2)[:200])
            raise AssistantError(-32002, f"{self.name}.{tool} failed: {text[:200]}")
        return res.structured_content

    async def call(self, tool: str, params: dict, out_model):
        return out_model.model_validate(await self.call_raw(tool, {"params": params}))
