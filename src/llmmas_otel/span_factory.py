from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, MutableMapping, Mapping
import hashlib
import uuid
import time
from contextvars import ContextVar

from opentelemetry import trace, propagate
from opentelemetry.trace import Span, SpanKind, Link

from . import semconv
from . import message_store


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_CURRENT_A2A_RECEIVE_DECISION: ContextVar[Optional[object]] = ContextVar(
    "llmmas_current_a2a_receive_decision", default=None
)
_CURRENT_TOOL_CALL_DECISION: ContextVar[Optional[object]] = ContextVar(
    "llmmas_current_tool_call_decision", default=None
)


@dataclass(frozen=True)
class A2ASendContext:
    span: Span
    decision: object
    effective_body: Optional[str]


@dataclass(frozen=True)
class ToolCallContext:
    span: Span
    decision: object
    call_id: str


class SpanFactory:
    def __init__(self, tracer_name: str = "llmmas-otel") -> None:
        self._tracer = trace.get_tracer(tracer_name)

    def current_a2a_receive_decision(self) -> Optional[object]:
        return _CURRENT_A2A_RECEIVE_DECISION.get()

    def current_tool_call_decision(self) -> Optional[object]:
        return _CURRENT_TOOL_CALL_DECISION.get()

    @contextmanager
    def session(self, *, session_id: str) -> Iterator[Span]:
        with message_store.session_context(session_id):
            with self._tracer.start_as_current_span(semconv.SPAN_SESSION) as span:
                span.set_attribute(semconv.ATTR_SESSION_ID, session_id)
                yield span

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
        apply_mutation: Optional[callable] = None,
    ) -> Iterator[A2ASendContext]:
        seg = message_store.current_segment() or {}
        session_id = message_store.current_session_id()

        decision = None
        effective_body = message_body
        original_sha: Optional[str] = None

        try:
            from .injection import HookContext, HookType, get_engine, is_enabled, DecisionKind
        except Exception:
            HookContext = None  # type: ignore
            HookType = None     # type: ignore
            get_engine = None   # type: ignore
            is_enabled = lambda: False  # type: ignore
            DecisionKind = None # type: ignore

        if is_enabled() and HookContext is not None:
            ctx = HookContext(
                hook_type=HookType.A2A_SEND,
                session_id=session_id,
                phase_name=seg.get("name"),
                phase_order=seg.get("order"),
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                edge_id=edge_id,
                message_id=message_id,
                channel=channel,
                agent_id=source_agent_id,
            )
            decision = get_engine().decide(ctx, payload=message_body)

            if decision.kind == DecisionKind.DELAY and decision.delay_ms is not None:
                time.sleep(decision.delay_ms / 1000.0)

            if decision.kind == DecisionKind.MUTATE:
                if message_body is None:
                    raise ValueError("Fault injection MUTATE requires message_body (string)")
                original_sha = _sha256_hex(message_body)
                effective_body = decision.mutated_payload
                if apply_mutation is not None and effective_body is not None:
                    try:
                        apply_mutation(effective_body)
                    except Exception:
                        pass

        span_name = f"{semconv.A2A_OP_SEND} {edge_id}"
        with self._tracer.start_as_current_span(span_name, kind=SpanKind.PRODUCER) as span:
            span.set_attribute(semconv.ATTR_SOURCE_AGENT_ID, source_agent_id)
            span.set_attribute(semconv.ATTR_TARGET_AGENT_ID, target_agent_id)
            span.set_attribute(semconv.ATTR_EDGE_ID, edge_id)
            span.set_attribute(semconv.ATTR_MESSAGE_ID, message_id)
            if channel is not None:
                span.set_attribute(semconv.ATTR_CHANNEL, channel)

            if decision is not None:
                try:
                    from .injection import DecisionKind
                    if decision.kind != DecisionKind.PASS:
                        span.set_attribute(semconv.ATTR_FAULT_INJECTED, True)
                        span.set_attribute(semconv.ATTR_FAULT_TYPE, decision.fault_type or "")
                        span.set_attribute(semconv.ATTR_FAULT_SPEC_ID, decision.fault_id or "")
                        span.set_attribute(semconv.ATTR_FAULT_DECISION, str(decision.kind.value))
                        span.add_event(
                            "fault.applied",
                            attributes={
                                semconv.ATTR_FAULT_SPEC_ID: decision.fault_id or "",
                                semconv.ATTR_FAULT_TYPE: decision.fault_type or "",
                                semconv.ATTR_FAULT_DECISION: str(decision.kind.value),
                                **(decision.metadata or {}),
                                **({"delay_ms": decision.delay_ms} if getattr(decision, "delay_ms", None) is not None else {}),
                            },
                        )
                except Exception:
                    pass

            if propagate_context and carrier is not None:
                propagate.inject(carrier)

            if effective_body is not None:
                preview = effective_body[:preview_chars]
                sha = _sha256_hex(effective_body)

                if message_store.is_enabled():
                    message_store.write_message(
                        direction="send",
                        message_id=message_id,
                        sha256=sha,
                        body=effective_body,
                        source_agent_id=source_agent_id,
                        target_agent_id=target_agent_id,
                        edge_id=edge_id,
                        channel=channel,
                        original_sha256=original_sha,
                        fault_spec_id=getattr(decision, "fault_id", None) if decision is not None else None,
                        fault_type=getattr(decision, "fault_type", None) if decision is not None else None,
                        fault_decision=str(getattr(decision, "kind", "pass")) if decision is not None else None,
                        dropped=(getattr(decision, "kind", None).value == "drop") if decision is not None and hasattr(getattr(decision, "kind", None), "value") else False,
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

            yield A2ASendContext(span=span, decision=decision, effective_body=effective_body)

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
        seg = message_store.current_segment() or {}
        session_id = message_store.current_session_id()

        links = None
        if link_from_carrier and carrier is not None:
            extracted_ctx = propagate.extract(carrier)
            extracted_sc = trace.get_current_span(extracted_ctx).get_span_context()
            if extracted_sc is not None and extracted_sc.is_valid:
                links = [Link(extracted_sc)]

        decision = None
        effective_body = message_body

        try:
            from .injection import HookContext, HookType, get_engine, is_enabled, DecisionKind
        except Exception:
            HookContext = None  # type: ignore
            HookType = None     # type: ignore
            get_engine = None   # type: ignore
            is_enabled = lambda: False  # type: ignore
            DecisionKind = None # type: ignore

        if is_enabled() and HookContext is not None:
            ctx = HookContext(
                hook_type=HookType.A2A_RECEIVE,
                session_id=session_id,
                phase_name=seg.get("name"),
                phase_order=seg.get("order"),
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                edge_id=edge_id,
                message_id=message_id,
                channel=channel,
                agent_id=target_agent_id,
            )
            decision = get_engine().decide(ctx, payload=message_body)

            if decision.kind == DecisionKind.DELAY and decision.delay_ms is not None:
                time.sleep(decision.delay_ms / 1000.0)

            if hasattr(decision, "kind") and getattr(decision.kind, "value", "") == "mutate":
                effective_body = getattr(decision, "mutated_payload", effective_body)

        token = _CURRENT_A2A_RECEIVE_DECISION.set(decision)

        span_name = f"{semconv.A2A_OP_PROCESS} {edge_id}"
        try:
            with self._tracer.start_as_current_span(span_name, kind=SpanKind.CONSUMER, links=links) as span:
                span.set_attribute(semconv.ATTR_SOURCE_AGENT_ID, source_agent_id)
                span.set_attribute(semconv.ATTR_TARGET_AGENT_ID, target_agent_id)
                span.set_attribute(semconv.ATTR_EDGE_ID, edge_id)
                span.set_attribute(semconv.ATTR_MESSAGE_ID, message_id)
                if channel is not None:
                    span.set_attribute(semconv.ATTR_CHANNEL, channel)

                if decision is not None:
                    try:
                        from .injection import DecisionKind
                        if decision.kind != DecisionKind.PASS:
                            span.set_attribute(semconv.ATTR_FAULT_INJECTED, True)
                            span.set_attribute(semconv.ATTR_FAULT_TYPE, decision.fault_type or "")
                            span.set_attribute(semconv.ATTR_FAULT_SPEC_ID, decision.fault_id or "")
                            span.set_attribute(semconv.ATTR_FAULT_DECISION, str(decision.kind.value))
                            span.add_event(
                                "fault.applied",
                                attributes={
                                    semconv.ATTR_FAULT_SPEC_ID: decision.fault_id or "",
                                    semconv.ATTR_FAULT_TYPE: decision.fault_type or "",
                                    semconv.ATTR_FAULT_DECISION: str(decision.kind.value),
                                    **(decision.metadata or {}),
                                    **({"delay_ms": decision.delay_ms} if getattr(decision, "delay_ms", None) is not None else {}),
                                },
                            )
                    except Exception:
                        pass

                if effective_body is not None:
                    preview = effective_body[:preview_chars]
                    sha = _sha256_hex(effective_body)

                    if message_store.is_enabled():
                        dropped = False
                        try:
                            from .injection import DecisionKind
                            dropped = decision is not None and decision.kind == DecisionKind.DROP
                        except Exception:
                            dropped = False

                        message_store.write_message(
                            direction="receive",
                            message_id=message_id,
                            sha256=sha,
                            body=effective_body,
                            source_agent_id=source_agent_id,
                            target_agent_id=target_agent_id,
                            edge_id=edge_id,
                            channel=channel,
                            fault_spec_id=getattr(decision, "fault_id", None) if decision is not None else None,
                            fault_type=getattr(decision, "fault_type", None) if decision is not None else None,
                            fault_decision=str(getattr(decision, "kind", "pass")) if decision is not None else None,
                            dropped=dropped,
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
        finally:
            _CURRENT_A2A_RECEIVE_DECISION.reset(token)

    @contextmanager
    def tool_call(
        self,
        *,
        tool_name: str,
        tool_call_id: Optional[str] = None,
        tool_type: Optional[str] = None,
        tool_args: Optional[str] = None,
        preview_chars: int = 200,
        record_args: bool = False,
    ) -> Iterator[ToolCallContext]:
        """
        Tool execution span + fault injection overlay hook.

        - Creates INTERNAL span: "execute_tool {tool_name}"
        - Applies injection decisions at TOOL_CALL:
            PASS / DELAY / RAISE / RETURN
        - Records llmmas.fault.* + fault.applied event on the tool span
        - Enforcement of RAISE/RETURN is done by the decorator using current_tool_call_decision().
        """
        seg = message_store.current_segment() or {}
        session_id = message_store.current_session_id()

        call_id = tool_call_id or f"toolcall-{uuid.uuid4().hex[:12]}"

        decision = None
        try:
            from .injection import HookContext, HookType, get_engine, is_enabled, DecisionKind
        except Exception:
            HookContext = None  # type: ignore
            HookType = None     # type: ignore
            get_engine = None   # type: ignore
            is_enabled = lambda: False  # type: ignore
            DecisionKind = None # type: ignore

        if is_enabled() and HookContext is not None:
            ctx = HookContext(
                hook_type=HookType.TOOL_CALL,
                session_id=session_id,
                phase_name=seg.get("name"),
                phase_order=seg.get("order"),
                tool_name=tool_name,
                tool_type=tool_type,
                tool_call_id=call_id,
            )
            decision = get_engine().decide(ctx, payload=tool_args)

            if decision.kind == DecisionKind.DELAY and decision.delay_ms is not None:
                time.sleep(decision.delay_ms / 1000.0)

        token = _CURRENT_TOOL_CALL_DECISION.set(decision)

        span_name = f"{semconv.GEN_AI_OPERATION_EXECUTE_TOOL} {tool_name}"
        try:
            with self._tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
                span.set_attribute(semconv.ATTR_GEN_AI_OPERATION_NAME, semconv.GEN_AI_OPERATION_EXECUTE_TOOL)
                span.set_attribute(semconv.ATTR_GEN_AI_TOOL_NAME, tool_name)
                span.set_attribute(semconv.ATTR_GEN_AI_TOOL_CALL_ID, call_id)
                if tool_type is not None:
                    span.set_attribute(semconv.ATTR_GEN_AI_TOOL_TYPE, tool_type)

                if record_args and tool_args is not None:
                    span.set_attribute(semconv.ATTR_TOOL_ARGS_PREVIEW, tool_args[:preview_chars])
                    span.set_attribute(semconv.ATTR_TOOL_ARGS_SHA256, _sha256_hex(tool_args))

                if decision is not None:
                    try:
                        from .injection import DecisionKind
                        if decision.kind != DecisionKind.PASS:
                            span.set_attribute(semconv.ATTR_FAULT_INJECTED, True)
                            span.set_attribute(semconv.ATTR_FAULT_TYPE, decision.fault_type or "")
                            span.set_attribute(semconv.ATTR_FAULT_SPEC_ID, decision.fault_id or "")
                            span.set_attribute(semconv.ATTR_FAULT_DECISION, str(decision.kind.value))
                            span.add_event(
                                "fault.applied",
                                attributes={
                                    semconv.ATTR_FAULT_SPEC_ID: decision.fault_id or "",
                                    semconv.ATTR_FAULT_TYPE: decision.fault_type or "",
                                    semconv.ATTR_FAULT_DECISION: str(decision.kind.value),
                                    **(decision.metadata or {}),
                                    **({"delay_ms": decision.delay_ms} if getattr(decision, "delay_ms", None) is not None else {}),
                                },
                            )
                    except Exception:
                        pass

                yield ToolCallContext(span=span, decision=decision, call_id=call_id)
        finally:
            _CURRENT_TOOL_CALL_DECISION.reset(token)


default_span_factory = SpanFactory()