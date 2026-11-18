"""Plugins. All six configured plugins are real. ragas_auditor is a lexical proxy (see quality.py), not RAGAS."""
from . import injection
from .pii import redact
from .specs import Verdict, hookimpl


class _PassThrough:
    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    @hookimpl
    def pre_execute(self, ctx):
        return Verdict()

    @hookimpl
    def before_tool(self, ctx, call):
        return Verdict()


class PiiRedactor:
    """Fail-closed (-32013): any redactor fault blocks the request. Replaces ctx['message'] with the redacted text."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    @hookimpl
    def pre_execute(self, ctx):
        try:
            text, counts = redact(ctx["message"])
        except Exception as exc:  # noqa: BLE001
            return Verdict.deny(-32013, f"redactor fault: {exc.__class__.__name__}")
        return Verdict(ctx_updates={"message": text, "pii_counts": counts})


class InjectionShield:
    """Layer 1: local regex heuristics (fast, but do not generalise to paraphrases — see evals).
    Layer 2 (opt-in, RA_INJECTION_MODEL): Prompt Guard classifier; fails open to layer 1 and records that it did."""

    def __init__(self, cfg: dict | None = None):
        from . import promptguard
        self.threshold = (cfg or {}).get("threshold", injection.THRESHOLD)
        self.pg = promptguard.from_env()
        self.pg_threshold = promptguard.THRESHOLD

    @hookimpl
    def pre_execute(self, ctx):
        sc, hits = injection.score(ctx["message"])
        if sc >= self.threshold:
            return Verdict.deny(-32014, f"injection score {sc}", score=sc)
        upd = {"injection_score": sc}
        if self.pg is not None:
            g = self.pg.score(ctx["message"])
            if g is None:
                upd["injection_classifier"] = "unavailable (heuristics only)"
            elif g >= self.pg_threshold:
                return Verdict.deny(-32014, f"classifier score {g:.3f}", score=g)
            else:
                upd["injection_classifier"] = round(g, 4)
        return Verdict(ctx_updates=upd)


class CircuitBreaker:
    """Trips when the same (tool, canonical_args_hash) is attempted more than `max_identical_calls` times in one
    request -> HALTED_BREAKER (-32032). Counts live in ctx['breaker_counts'], owned by the loop, not the LLM.
    Operators can also force a tool open (trip/reset) from the workbench; every denial is kept for inspection."""

    def __init__(self, cfg: dict | None = None):
        from collections import deque
        self.max_identical = (cfg or {}).get("max_identical_calls", 2)
        self.forced: set[str] = set()
        self.events: deque = deque(maxlen=100)

    def trip(self, tool: str) -> None:
        self.forced.add(tool)

    def reset(self, tool: str | None = None) -> None:
        self.forced.clear() if tool is None else self.forced.discard(tool)

    @hookimpl
    def before_tool(self, ctx, call):
        import time
        if call["tool"] in self.forced:
            self.events.append({"ts": time.time(), "tool": call["tool"], "reason": "manually tripped"})
            return Verdict.deny(-32032, f"{call['tool']} breaker manually tripped")
        counts = ctx.setdefault("breaker_counts", {})
        key = (call["tool"], call["args_hash"])
        counts[key] = counts.get(key, 0) + 1
        if counts[key] > self.max_identical:
            self.events.append({"ts": time.time(), "tool": call["tool"], "args_hash": call["args_hash"],
                                "reason": f"identical call #{counts[key]}"})
            return Verdict.deny(-32032, f"{call['tool']} repeated {counts[key] - 1}x with identical args; "
                                        "change the arguments or answer from gathered evidence")
        return Verdict()


class Hitl:
    """Policy: any Tier 3 call is gated (-32015). The harness then records PENDING_APPROVAL via mcp-state-store;
    the action itself is never executed. KPI-19 = 100 %."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    @hookimpl
    def before_tool(self, ctx, call):
        if call["tier"] == "TIER_3_MUTATE":
            return Verdict.deny(-32015, "awaiting human approval")
        return Verdict()


class RagasAuditor:
    """POST_EXECUTION audit (lexical proxy, see quality.py). Never blocks the response; flags + gates the cache write."""

    def __init__(self, cfg: dict | None = None):
        from ..quality import RagasAuditorImpl
        self.impl = RagasAuditorImpl(cfg)

    @hookimpl
    def post_execute(self, ctx, answer):
        self.impl.run(ctx, answer)


class OtelExporter:
    def __init__(self, cfg: dict | None = None):
        from ..telemetry import OtelExporter as Impl
        self.impl = Impl(cfg)

    @hookimpl
    def on_span(self, span):
        self.impl.on_span_impl(span)


DEFAULTS = {"pii_redactor": PiiRedactor, "injection_shield": InjectionShield,
            "circuit_breaker": CircuitBreaker, "hitl": Hitl,
            "ragas_auditor": RagasAuditor, "otel_exporter": OtelExporter}
