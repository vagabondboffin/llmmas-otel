from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional
import hashlib

from opentelemetry import trace
from opentelemetry.trace import Span

from . import semconv

class SpanFactory:
    """
    I am not sure about the name yet but the idea is to keep things centralized.
    Decorators and future framework adapters should call this, not create spans directly.
    """

    def __init__(self, tracer_name: str = "llmmas-otel") -> None:
        self._tracer = trace.get_tracer(tracer_name)

    @contextmanager
    def session(self, *, session_id: str) -> Iterator[Span]:
        with self._tracer.start_as_current_span(semconv.SPAN_SESSION) as span:
            span.set_attribute(semconv.ATTR_SESSION_ID, session_id)
            yield span

    @contextmanager
    def agent_step(self, *, agent_id: str, step_index: int) -> Iterator[Span]:
        with self._tracer.start_as_current_span(semconv.SPAN_AGENT_STEP) as span:
            span.set_attribute(semconv.ATTR_AGENT_ID, agent_id)
            span.set_attribute(semconv.ATTR_STEP_INDEX, step_index)
            yield span

    @contextmanager
    def a2a_send(
        self,
        *,
        source_agent_id: str,
        target_agent_id: str,
        edge_id: str,
        message_id: str,
        channel: Optional[str] = None,
            message_body: Optional[str] = None,
            preview_chars: int = 200,
            add_event: bool = True,
    ) -> Iterator[Span]:
        """
        A2A send span.
        Minimal "message observability":
          - preview (truncated) as attribute
          - sha256 hash as attribute
          - optional span event for easy viewing in Jaeger
        """
        with self._tracer.start_as_current_span(semconv.SPAN_A2A_CLIENT) as span:
            # IDs / topology-ish correlation
            span.set_attribute(semconv.ATTR_SOURCE_AGENT_ID, source_agent_id)
            span.set_attribute(semconv.ATTR_TARGET_AGENT_ID, target_agent_id)
            span.set_attribute(semconv.ATTR_EDGE_ID, edge_id)
            span.set_attribute(semconv.ATTR_MESSAGE_ID, message_id)

            if channel is not None:
                span.set_attribute("llmmas.channel", channel)

            # Content-related fields (safe defaults)
            if message_body is not None:
                preview = message_body[:preview_chars]
                sha = hashlib.sha256(message_body.encode("utf-8")).hexdigest()

                span.set_attribute(semconv.ATTR_MESSAGE_PREVIEW, preview)
                span.set_attribute(semconv.ATTR_MESSAGE_SHA256, sha)

                if add_event:
                    span.add_event(
                        "a2a.message",
                        attributes={
                            "llmmas.message.id": message_id,
                            "llmmas.message.preview": preview,
                            "llmmas.message.sha256": sha,
                        },
                    )

            yield span

default_span_factory = SpanFactory()
