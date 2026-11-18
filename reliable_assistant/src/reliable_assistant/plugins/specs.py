"""Pluggy hook specs and Verdict — design.md Appendix A."""
from dataclasses import dataclass, field
from typing import Any

import pluggy

PROJECT = "unreliable_assistant"  # entry-point group: unreliable_assistant.plugins
hookspec = pluggy.HookspecMarker(PROJECT)
hookimpl = pluggy.HookimplMarker(PROJECT)


@dataclass
class Verdict:
    allow: bool = True
    code: int | None = None          # error code from errors.CODES when denied
    reason: str = ""
    ctx_updates: dict[str, Any] = field(default_factory=dict)  # e.g. {"message": redacted}
    data: dict[str, Any] = field(default_factory=dict)         # propagated into AssistantError.data on deny

    @classmethod
    def deny(cls, code: int, reason: str = "", **data) -> "Verdict":
        return cls(allow=False, code=code, reason=reason, data=data)


class Specs:
    @hookspec
    def pre_execute(self, ctx: dict) -> Verdict | None:
        """PRE_EXECUTION: pii_redactor(10), injection_shield(20). Fail-closed."""

    @hookspec
    def before_tool(self, ctx: dict, call: dict) -> Verdict | None:
        """TOOL_DISPATCH: circuit_breaker(10), hitl(20). Fail-closed."""

    @hookspec
    def post_execute(self, ctx: dict, answer: dict) -> None:
        """POST_EXECUTION: ragas_auditor(10). Async, never blocks the response."""

    @hookspec
    def on_span(self, span: dict) -> None:
        """ALL_SPANS: otel_exporter. Fail-open."""
