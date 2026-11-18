"""Tracing (design.md Appendix D). Requests record plain span dicts; the `otel_exporter` plugin (ALL_SPANS, fail-open)
turns them into OpenTelemetry spans. Spans are kept in a bounded in-memory ring (for Streamlit / tests) and are sent
over OTLP ONLY when RA_OTLP_ENDPOINT is set — the doc's default :4317 is occupied on this host by another project's
SigNoz, so nothing is exported there implicitly."""
import os
import secrets
import time
from collections import deque
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, set_span_in_context


class RingExporter(SpanExporter):
    def __init__(self, maxlen: int = 5000):
        self.spans: deque[ReadableSpan] = deque(maxlen=maxlen)

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


RING = RingExporter()
_provider = TracerProvider(resource=Resource.create({"service.name": "reliable-assistant"}))
_provider.add_span_processor(SimpleSpanProcessor(RING))
if os.environ.get("RA_OTLP_ENDPOINT"):
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=os.environ["RA_OTLP_ENDPOINT"], insecure=True)))
_tracer = _provider.get_tracer("reliable_assistant")


def spans_for(trace_id: str) -> list[ReadableSpan]:
    return [s for s in RING.spans if format(s.context.trace_id, "032x") == trace_id]


class Recorder:
    """Per-request span collector. Spans are flushed (in start order) to the ALL_SPANS plugin hook at request end."""

    def __init__(self, trace_id: str):
        self.trace_id, self.spans, self._stack = trace_id, [], []

    @contextmanager
    def span(self, name: str, **attrs):
        d = {"name": name, "trace_id": self.trace_id, "span_id": secrets.token_hex(8),
             "parent_id": self._stack[-1] if self._stack else None, "start_ns": time.time_ns(), "attrs": dict(attrs)}
        self._stack.append(d["span_id"])
        try:
            yield d["attrs"]
        except BaseException as exc:
            d["attrs"]["error"] = exc.__class__.__name__
            raise
        finally:
            self._stack.pop()
            d["end_ns"] = time.time_ns()
            self.spans.append(d)

    def add(self, name: str, start_ns: int, duration_ms: float, parent_id: str | None = None, **attrs):
        """Add an already-measured child span (e.g. server-reported stage timings)."""
        self.spans.append({"name": name, "trace_id": self.trace_id, "span_id": secrets.token_hex(8),
                           "parent_id": parent_id or (self._stack[-1] if self._stack else None),
                           "start_ns": start_ns, "end_ns": start_ns + int(duration_ms * 1e6), "attrs": attrs})

    def flush(self, pm) -> None:
        for d in sorted(self.spans, key=lambda x: x["start_ns"]):
            pm.run_observers("ALL_SPANS", span=d)      # fail-open: telemetry must never break a request


class OtelExporter:
    """`otel_exporter` plugin implementation (registered in plugins/builtin)."""

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg
        self._real: dict[str, trace.Span] = {}

    def on_span_impl(self, d: dict) -> None:
        tid = int(d["trace_id"], 16)
        parent = self._real.get(d["parent_id"]) if d["parent_id"] else None
        if parent is not None:
            ctx = set_span_in_context(parent)
        else:   # root: attach to the request's trace id as if continuing an upstream W3C traceparent
            ctx = set_span_in_context(NonRecordingSpan(SpanContext(tid, int(secrets.token_hex(8), 16), True, TraceFlags(1))))
        sp = _tracer.start_span(d["name"], context=ctx, start_time=d["start_ns"], attributes={
            k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in d["attrs"].items()})
        self._real[d["span_id"]] = sp
        sp.end(end_time=d["end_ns"])
        if len(self._real) > 2000:
            self._real.clear()
