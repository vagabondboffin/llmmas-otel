from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, MutableMapping, Mapping
import hashlib
import uuid

from opentelemetry import trace, propagate
from opentelemetry.trace import Span, SpanKind, Link

from . import semconv
from . import message_store


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SpanFactory:
    """
    Central place for creating spans that correspond to the proposal.

    Decorators and future framework adapters should call this, not create spans directly.
    """

    def __init__(self, tracer_name: str = "llmmas-otel") -> None:
        self._tracer = trace.get_tracer(tracer_name)

    @contextmanager
    def session(self, *, session_id: str) -> Iterator[Span]:
        with message_store.session_context(session_id):
            with self._tracer.start_as_current_span(semconv.SPAN_SESSION) as span:
                span.set_attribute(semconv.ATTR_SESSION_ID, session_id)
                yield span

    # "segment" maps to proposal's workflow segment/phase concept.
    @contextmanager
    def segment(
        self,
        *,
        name: str,
        order: int = 0,
        origin: Optional[str] = None,
    ) -> Iterator[Span]:
        with message_store.segment_context(name=name, order=order):
            with self._tracer.start_as_current_span(semconv.SPAN_SEGMENT) as span:
                span.set_attribute(semconv.ATTR_SEGMENT_NAME, name)
                span.set_attribute(semconv.ATTR_SEGMENT_ORDER, order)
                if origin is not None:
                    span.set_attribute(semconv.ATTR_SEGMENT_ORIGIN, origin)
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
        carrier: Optional[MutableMapping[str, str]] = None,
        propagate_context: bool = True,
        preview_chars: int = 200,
        add_event: bool = True,
    ) -> Iterator[Span]:
        """
        A2A send span (producer side).

        - Creates a PRODUCER span named: "send {edge_id}"
        - Optionally injects trace context into a provided carrier mapping
        - Adds preview/hash (and optional event) for lightweight message observability
        - Writes full body to message_store (if enabled)
        """
        span_name = f"{semconv.A2A_OP_SEND} {edge_id}"

        with self._tracer.start_as_current_span(span_name, kind=SpanKind.PRODUCER) as span:
            # IDs / topology-ish correlation
            span.set_attribute(semconv.ATTR_SOURCE_AGENT_ID, source_agent_id)
            span.set_attribute(semconv.ATTR_TARGET_AGENT_ID, target_agent_id)
            span.set_attribute(semconv.ATTR_EDGE_ID, edge_id)
            span.set_attribute(semconv.ATTR_MESSAGE_ID, message_id)
            if channel is not None:
                span.set_attribute(semconv.ATTR_CHANNEL, channel)

            # Propagate message context for receiver correlation
            if propagate_context and carrier is not None:
                propagate.inject(carrier)

            if message_body is not None:
                preview = message_body[:preview_chars]
                sha = _sha256_hex(message_body)

                if message_store.is_enabled():
                    message_store.write_message(
                        direction="send",
                        message_id=message_id,
                        sha256=sha,
                        body=message_body,
                        source_agent_id=source_agent_id,
                        target_agent_id=target_agent_id,
                        edge_id=edge_id,
                        channel=channel,
                    )

                span.set_attribute(semconv.ATTR_MESSAGE_PREVIEW, preview)
                span.set_attribute(semconv.ATTR_MESSAGE_SHA256, sha)

                if add_event:
                    span.add_event(
                        "a2a.message",
                        attributes={
                            semconv.ATTR_MESSAGE_ID: message_id,
                            semconv.ATTR_MESSAGE_PREVIEW: preview,
                            semconv.ATTR_MESSAGE_SHA256: sha,
                            "llmmas.message.direction": "send",
                        },
                    )

            yield span

    @contextmanager
    def a2a_receive(
        self,
        *,
        source_agent_id: str,
        target_agent_id: str,
        edge_id: str,
        message_id: str,
        channel: Optional[str] = None,
        message_body: Optional[str] = None,
        carrier: Optional[Mapping[str, str]] = None,
        link_from_carrier: bool = True,
        preview_chars: int = 200,
        add_event: bool = True,
    ) -> Iterator[Span]:
        """
        A2A receive/process span (consumer side).

        - Creates a CONSUMER span named: "process {edge_id}"
        - By default links to the extracted message context (if carrier provided)
          rather than parenting from it. This preserves the proposal hierarchy:
            session -> segment -> agent_step -> process
        """

        links = None
        if link_from_carrier and carrier is not None:
            extracted_ctx = propagate.extract(carrier)
            extracted_sc = trace.get_current_span(extracted_ctx).get_span_context()
            if extracted_sc is not None and extracted_sc.is_valid:
                links = [Link(extracted_sc)]

        span_name = f"{semconv.A2A_OP_PROCESS} {edge_id}"
        with self._tracer.start_as_current_span(
            span_name,
            kind=SpanKind.CONSUMER,
            links=links,
        ) as span:
            span.set_attribute(semconv.ATTR_SOURCE_AGENT_ID, source_agent_id)
            span.set_attribute(semconv.ATTR_TARGET_AGENT_ID, target_agent_id)
            span.set_attribute(semconv.ATTR_EDGE_ID, edge_id)
            span.set_attribute(semconv.ATTR_MESSAGE_ID, message_id)
            if channel is not None:
                span.set_attribute(semconv.ATTR_CHANNEL, channel)

            if message_body is not None:
                preview = message_body[:preview_chars]
                sha = _sha256_hex(message_body)

                if message_store.is_enabled():
                    message_store.write_message(
                        direction="receive",
                        message_id=message_id,
                        sha256=sha,
                        body=message_body,
                        source_agent_id=source_agent_id,
                        target_agent_id=target_agent_id,
                        edge_id=edge_id,
                        channel=channel,
                    )

                span.set_attribute(semconv.ATTR_MESSAGE_PREVIEW, preview)
                span.set_attribute(semconv.ATTR_MESSAGE_SHA256, sha)

                if add_event:
                    span.add_event(
                        "a2a.message",
                        attributes={
                            semconv.ATTR_MESSAGE_ID: message_id,
                            semconv.ATTR_MESSAGE_PREVIEW: preview,
                            semconv.ATTR_MESSAGE_SHA256: sha,
                            "llmmas.message.direction": "receive",
                        },
                    )

            yield span

    @contextmanager
    def tool_call(
        self,
        *,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        tool_type: Optional[str] = None,
        tool_args: Optional[str] = None,
        tool_result: Optional[str] = None,
        preview_chars: int = 200,
        record_args: bool = False,
        record_result: bool = False,
    ) -> Iterator[Span]:
        """
        Tool execution span.

        We follow GenAI semconv for the core identity fields:
          - gen_ai.operation.name = "execute_tool"
          - gen_ai.tool.name
          - gen_ai.tool.type (optional)
          - gen_ai.tool.call.id


        """
        call_id = tool_call_id or f"toolcall-{uuid.uuid4().hex[:12]}"
        span_name = f"{semconv.GEN_AI_OPERATION_EXECUTE_TOOL} {tool_name}"

        with self._tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
            span.set_attribute(semconv.ATTR_GEN_AI_OPERATION_NAME, semconv.GEN_AI_OPERATION_EXECUTE_TOOL)
            span.set_attribute(semconv.ATTR_GEN_AI_TOOL_NAME, tool_name)
            span.set_attribute(semconv.ATTR_GEN_AI_TOOL_CALL_ID, call_id)
            if tool_type is not None:
                span.set_attribute(semconv.ATTR_GEN_AI_TOOL_TYPE, tool_type)

            if record_args and tool_args is not None:
                span.set_attribute(semconv.ATTR_TOOL_ARGS_PREVIEW, tool_args[:preview_chars])
                span.set_attribute(semconv.ATTR_TOOL_ARGS_SHA256, _sha256_hex(tool_args))

            if record_result and tool_result is not None:
                span.set_attribute(semconv.ATTR_TOOL_RESULT_PREVIEW, tool_result[:preview_chars])
                span.set_attribute(semconv.ATTR_TOOL_RESULT_SHA256, _sha256_hex(tool_result))

            yield span


default_span_factory = SpanFactory()
