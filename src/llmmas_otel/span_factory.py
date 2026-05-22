from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, MutableMapping, Optional
import hashlib
import os
import time
import uuid

from opentelemetry import propagate, trace
from opentelemetry.trace import Link, Span, SpanKind

from . import message_store, semconv


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _coerce_attr_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, bool, int, float)) for item in value
    ):
        return list(value)
    return str(value)


def _set_attr(span: Span, key: str, value: Any) -> None:
    coerced = _coerce_attr_value(value)
    if coerced is not None:
        span.set_attribute(key, coerced)


def _set_metadata(span: Span, metadata: Optional[Mapping[str, Any]], prefix: str) -> None:
    if not metadata:
        return
    for key, value in metadata.items():
        safe_key = str(key).replace(" ", "_")
        _set_attr(span, f"{prefix}.{safe_key}", value)


_CURRENT_A2A_RECEIVE_DECISION: ContextVar[Optional[object]] = ContextVar(
    "llmmas_current_a2a_receive_decision",
    default=None,
)
_CURRENT_TOOL_CALL_DECISION: ContextVar[Optional[object]] = ContextVar(
    "llmmas_current_tool_call_decision",
    default=None,
)
_CURRENT_LLM_CALL_DECISION: ContextVar[Optional[object]] = ContextVar(
    "llmmas_current_llm_call_decision",
    default=None,
)


@dataclass(frozen=True)
class A2ASendContext:
    span: Span
    decision: Optional[object]
    effective_body: Optional[str]


@dataclass(frozen=True)
class ToolCallContext:
    span: Span
    decision: Optional[object]
    call_id: str


@dataclass(frozen=True)
class LLMCallContext:
    span: Span
    decision: Optional[object]
    request_id: str


@dataclass(frozen=True)
class EnvironmentActionContext:
    span: Span
    decision: Optional[object]
    action_id: str


@dataclass(frozen=True)
class DelegationContext:
    span: Span
    delegation_id: str


@dataclass(frozen=True)
class ArtifactContext:
    span: Span
    artifact_id: str


def _fault_trace_visibility_enabled() -> bool:
    try:
        from .injection import is_fault_trace_visible

        return is_fault_trace_visible()
    except Exception:
        return True


def _annotate_fault_on_span(span: Span, decision: Optional[object]) -> None:
    if decision is None or not _fault_trace_visibility_enabled():
        return

    try:
        from .injection import DecisionKind
    except Exception:
        return

    if getattr(decision, "kind", None) == DecisionKind.PASS:
        return

    span.set_attribute(semconv.ATTR_FAULT_INJECTED, True)
    span.set_attribute(semconv.ATTR_FAULT_TYPE, getattr(decision, "fault_type", None) or "")
    span.set_attribute(semconv.ATTR_FAULT_SPEC_ID, getattr(decision, "fault_id", None) or "")
    span.set_attribute(
        semconv.ATTR_FAULT_DECISION,
        str(getattr(getattr(decision, "kind", None), "value", "")),
    )
    span.add_event(
        "fault.applied",
        attributes={
            semconv.ATTR_FAULT_SPEC_ID: getattr(decision, "fault_id", None) or "",
            semconv.ATTR_FAULT_TYPE: getattr(decision, "fault_type", None) or "",
            semconv.ATTR_FAULT_DECISION: str(
                getattr(getattr(decision, "kind", None), "value", "")
            ),
            **(getattr(decision, "metadata", None) or {}),
            **(
                {"delay_ms": getattr(decision, "delay_ms", None)}
                if getattr(decision, "delay_ms", None) is not None
                else {}
            ),
        },
    )


class SpanFactory:
    def __init__(self, tracer_name: str = "llmmas-otel") -> None:
        self._tracer = trace.get_tracer(tracer_name)

    def current_a2a_receive_decision(self) -> Optional[object]:
        return _CURRENT_A2A_RECEIVE_DECISION.get()

    def current_tool_call_decision(self) -> Optional[object]:
        return _CURRENT_TOOL_CALL_DECISION.get()

    def current_llm_call_decision(self) -> Optional[object]:
        return _CURRENT_LLM_CALL_DECISION.get()

    @contextmanager
    def session(
        self,
        *,
        session_id: str,
        name: Optional[str] = None,
        task_id: Optional[str] = None,
        framework: Optional[str] = None,
        system: Optional[str] = None,
        adapter: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[Span]:
        with message_store.session_context(session_id):
            span_name = semconv.SPAN_SESSION if name is None else f"{semconv.SPAN_SESSION} {name}"
            with self._tracer.start_as_current_span(span_name) as span:
                span.set_attribute(semconv.ATTR_SESSION_ID, session_id)
                _set_attr(span, semconv.ATTR_SESSION_NAME, name)
                _set_attr(span, semconv.ATTR_SESSION_TASK_ID, task_id)
                _set_attr(span, semconv.ATTR_FRAMEWORK, framework)
                _set_attr(span, semconv.ATTR_SYSTEM, system)
                _set_attr(span, semconv.ATTR_ADAPTER, adapter)
                _set_metadata(span, metadata, "llmmas.session.meta")
                yield span

    @contextmanager
    def workflow(
        self,
        *,
        name: str,
        order: int = 0,
        kind: str = "workflow",
        origin: Optional[str] = None,
        workflow_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[Span]:
        current = message_store.current_workflow()
        if parent_id is None and current is not None:
            parent_id = current.get("id")
        wid = workflow_id or f"workflow-{uuid.uuid4().hex[:12]}"
        depth = len(message_store.current_workflow_stack())

        with message_store.workflow_context(
            workflow_id=wid,
            name=name,
            order=order,
            kind=kind,
            origin=origin,
            parent_id=parent_id,
        ):
            span_name = f"{semconv.SPAN_WORKFLOW} {name}"
            with self._tracer.start_as_current_span(span_name) as span:
                span.set_attribute(semconv.ATTR_WORKFLOW_ID, wid)
                span.set_attribute(semconv.ATTR_WORKFLOW_NAME, name)
                span.set_attribute(semconv.ATTR_WORKFLOW_KIND, kind)
                span.set_attribute(semconv.ATTR_WORKFLOW_ORDER, order)
                span.set_attribute(semconv.ATTR_WORKFLOW_DEPTH, depth)
                _set_attr(span, semconv.ATTR_WORKFLOW_ORIGIN, origin)
                _set_attr(span, semconv.ATTR_WORKFLOW_PARENT_ID, parent_id)

                # Backward-compatible segment attributes for existing analysis code.
                span.set_attribute(semconv.ATTR_SEGMENT_NAME, name)
                span.set_attribute(semconv.ATTR_SEGMENT_ORDER, order)
                _set_attr(span, semconv.ATTR_SEGMENT_ORIGIN, origin)

                _set_metadata(span, metadata, "llmmas.workflow.meta")
                yield span

    @contextmanager
    def segment(
        self,
        *,
        name: str,
        order: int = 0,
        origin: Optional[str] = None,
        workflow_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        kind: str = "segment",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[Span]:
        with self.workflow(
            name=name,
            order=order,
            kind=kind,
            origin=origin,
            workflow_id=workflow_id,
            parent_id=parent_id,
            metadata=metadata,
        ) as span:
            yield span

    @contextmanager
    def agent_step(
        self,
        *,
        agent_id: str,
        step_index: int,
        agent_role: Optional[str] = None,
        agent_impl: Optional[str] = None,
        parent_agent_id: Optional[str] = None,
        step_kind: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[Span]:
        span_name = f"{semconv.SPAN_AGENT_STEP} {agent_id}"
        with self._tracer.start_as_current_span(span_name) as span:
            span.set_attribute(semconv.ATTR_AGENT_ID, agent_id)
            span.set_attribute(semconv.ATTR_STEP_INDEX, step_index)
            _set_attr(span, semconv.ATTR_AGENT_ROLE, agent_role)
            _set_attr(span, semconv.ATTR_AGENT_IMPL, agent_impl)
            _set_attr(span, semconv.ATTR_PARENT_AGENT_ID, parent_agent_id)
            _set_attr(span, semconv.ATTR_STEP_KIND, step_kind)
            _set_metadata(span, metadata, "llmmas.agent.meta")
            yield span

    @contextmanager
    def delegation(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        delegation_id: Optional[str] = None,
        task_id: Optional[str] = None,
        kind: str = "delegation",
        via: Optional[str] = None,
        goal: Optional[str] = None,
        preview_chars: int = 200,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[DelegationContext]:
        did = delegation_id or f"delegation-{uuid.uuid4().hex[:12]}"
        span_name = f"{semconv.SPAN_DELEGATION} {from_agent_id}->{to_agent_id}"
        with self._tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
            span.set_attribute(semconv.ATTR_DELEGATION_ID, did)
            span.set_attribute(semconv.ATTR_DELEGATION_FROM_AGENT, from_agent_id)
            span.set_attribute(semconv.ATTR_DELEGATION_TO_AGENT, to_agent_id)
            span.set_attribute(semconv.ATTR_DELEGATION_KIND, kind)
            _set_attr(span, semconv.ATTR_DELEGATION_TASK_ID, task_id)
            _set_attr(span, semconv.ATTR_DELEGATION_VIA, via)
            if goal is not None:
                span.set_attribute(semconv.ATTR_DELEGATION_GOAL_PREVIEW, goal[:preview_chars])
                span.set_attribute(semconv.ATTR_DELEGATION_GOAL_SHA256, _sha256_hex(goal))
            _set_metadata(span, metadata, "llmmas.delegation.meta")
            yield DelegationContext(span=span, delegation_id=did)

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
        apply_mutation: Optional[Callable[[str], None]] = None,
        route_via: Optional[str] = None,
        message_kind: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[A2ASendContext]:
        seg = message_store.current_segment() or {}
        session_id = message_store.current_session_id()

        decision = None
        effective_body = message_body
        original_sha: Optional[str] = None

        try:
            from .injection import DecisionKind, HookContext, HookType, get_engine, is_enabled
        except Exception:
            HookContext = None
            HookType = None
            get_engine = None
            is_enabled = lambda: False
            DecisionKind = None

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
                    apply_mutation(effective_body)

        span_name = f"{semconv.A2A_OP_SEND} {edge_id}"
        with self._tracer.start_as_current_span(span_name, kind=SpanKind.PRODUCER) as span:
            span.set_attribute(semconv.ATTR_SOURCE_AGENT_ID, source_agent_id)
            span.set_attribute(semconv.ATTR_TARGET_AGENT_ID, target_agent_id)
            span.set_attribute(semconv.ATTR_EDGE_ID, edge_id)
            span.set_attribute(semconv.ATTR_MESSAGE_ID, message_id)
            _set_attr(span, semconv.ATTR_CHANNEL, channel)
            _set_attr(span, semconv.ATTR_MESSAGE_ROUTE_VIA, route_via)
            _set_attr(span, semconv.ATTR_MESSAGE_KIND, message_kind)
            _set_attr(span, semconv.ATTR_MESSAGE_PARENT_ID, parent_message_id)
            _set_metadata(span, metadata, "llmmas.message.meta")

            _annotate_fault_on_span(span, decision)

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
                        message_kind=message_kind,
                        route_via=route_via,
                        parent_message_id=parent_message_id,
                        original_sha256=original_sha,
                        fault_spec_id=getattr(decision, "fault_id", None)
                        if decision is not None
                        else None,
                        fault_type=getattr(decision, "fault_type", None)
                        if decision is not None
                        else None,
                        fault_decision=str(getattr(decision, "kind", "pass"))
                        if decision is not None
                        else None,
                        dropped=(
                            getattr(decision, "kind", None).value == "drop"
                            if decision is not None
                            and hasattr(getattr(decision, "kind", None), "value")
                            else False
                        ),
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
                            semconv.ATTR_MESSAGE_DIRECTION: "send",
                            **({semconv.ATTR_MESSAGE_KIND: message_kind} if message_kind else {}),
                            **({semconv.ATTR_MESSAGE_ROUTE_VIA: route_via} if route_via else {}),
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
        route_via: Optional[str] = None,
        message_kind: Optional[str] = None,
        parent_message_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
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
            from .injection import DecisionKind, HookContext, HookType, get_engine, is_enabled
        except Exception:
            HookContext = None
            HookType = None
            get_engine = None
            is_enabled = lambda: False
            DecisionKind = None

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

            if decision.kind == DecisionKind.MUTATE:
                if message_body is None:
                    raise ValueError("Fault injection MUTATE requires message_body (string)")
                effective_body = decision.mutated_payload

        token = _CURRENT_A2A_RECEIVE_DECISION.set(decision)

        span_name = f"{semconv.A2A_OP_PROCESS} {edge_id}"
        try:
            with self._tracer.start_as_current_span(
                span_name,
                kind=SpanKind.CONSUMER,
                links=links,
            ) as span:
                span.set_attribute(semconv.ATTR_SOURCE_AGENT_ID, source_agent_id)
                span.set_attribute(semconv.ATTR_TARGET_AGENT_ID, target_agent_id)
                span.set_attribute(semconv.ATTR_EDGE_ID, edge_id)
                span.set_attribute(semconv.ATTR_MESSAGE_ID, message_id)
                _set_attr(span, semconv.ATTR_CHANNEL, channel)
                _set_attr(span, semconv.ATTR_MESSAGE_ROUTE_VIA, route_via)
                _set_attr(span, semconv.ATTR_MESSAGE_KIND, message_kind)
                _set_attr(span, semconv.ATTR_MESSAGE_PARENT_ID, parent_message_id)
                _set_metadata(span, metadata, "llmmas.message.meta")

                _annotate_fault_on_span(span, decision)

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
                            message_kind=message_kind,
                            route_via=route_via,
                            parent_message_id=parent_message_id,
                            fault_spec_id=getattr(decision, "fault_id", None)
                            if decision is not None
                            else None,
                            fault_type=getattr(decision, "fault_type", None)
                            if decision is not None
                            else None,
                            fault_decision=str(getattr(decision, "kind", "pass"))
                            if decision is not None
                            else None,
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
                                semconv.ATTR_MESSAGE_DIRECTION: "receive",
                                **({semconv.ATTR_MESSAGE_KIND: message_kind} if message_kind else {}),
                                **({semconv.ATTR_MESSAGE_ROUTE_VIA: route_via} if route_via else {}),
                            },
                        )

                yield span
        finally:
            _CURRENT_A2A_RECEIVE_DECISION.reset(token)

    @contextmanager
    def environment_action(
        self,
        *,
        name: str,
        kind: str = "tool",
        action_id: Optional[str] = None,
        input_text: Optional[str] = None,
        preview_chars: int = 200,
        record_input: bool = False,
        tool_type: Optional[str] = None,
        changed_files: Optional[list[str]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[EnvironmentActionContext]:
        seg = message_store.current_segment() or {}
        session_id = message_store.current_session_id()
        aid = action_id or f"envact-{uuid.uuid4().hex[:12]}"

        decision = None
        try:
            from .injection import DecisionKind, HookContext, HookType, get_engine, is_enabled
        except Exception:
            HookContext = None
            HookType = None
            get_engine = None
            is_enabled = lambda: False
            DecisionKind = None

        if is_enabled() and HookContext is not None:
            ctx = HookContext(
                hook_type=HookType.TOOL_CALL,
                session_id=session_id,
                phase_name=seg.get("name"),
                phase_order=seg.get("order"),
                tool_name=name,
                tool_type=tool_type or kind,
                tool_call_id=aid,
            )
            decision = get_engine().decide(ctx, payload=input_text)

            if decision.kind == DecisionKind.DELAY and decision.delay_ms is not None:
                time.sleep(decision.delay_ms / 1000.0)

        token = _CURRENT_TOOL_CALL_DECISION.set(decision)
        span_name = f"{semconv.SPAN_ENVIRONMENT_ACTION} {kind}:{name}"
        try:
            with self._tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
                span.set_attribute(semconv.ATTR_ENV_ACTION_ID, aid)
                span.set_attribute(semconv.ATTR_ENV_ACTION_KIND, kind)
                span.set_attribute(semconv.ATTR_ENV_ACTION_NAME, name)
                if changed_files:
                    span.set_attribute(semconv.ATTR_ENV_ACTION_CHANGED_FILES, changed_files)
                _set_metadata(span, metadata, "llmmas.env_action.meta")

                if record_input and input_text is not None:
                    span.set_attribute(semconv.ATTR_ENV_ACTION_INPUT_PREVIEW, input_text[:preview_chars])
                    span.set_attribute(semconv.ATTR_ENV_ACTION_INPUT_SHA256, _sha256_hex(input_text))

                _annotate_fault_on_span(span, decision)
                yield EnvironmentActionContext(span=span, decision=decision, action_id=aid)
        finally:
            _CURRENT_TOOL_CALL_DECISION.reset(token)

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
        with self.environment_action(
            name=tool_name,
            kind="tool",
            action_id=tool_call_id,
            input_text=tool_args,
            preview_chars=preview_chars,
            record_input=record_args,
            tool_type=tool_type,
        ) as env_ctx:
            span = env_ctx.span
            call_id = env_ctx.action_id
            span.set_attribute(semconv.ATTR_GEN_AI_OPERATION_NAME, semconv.GEN_AI_OPERATION_EXECUTE_TOOL)
            span.set_attribute(semconv.ATTR_GEN_AI_TOOL_NAME, tool_name)
            span.set_attribute(semconv.ATTR_GEN_AI_TOOL_CALL_ID, call_id)
            if tool_type is not None:
                span.set_attribute(semconv.ATTR_GEN_AI_TOOL_TYPE, tool_type)
            if record_args and tool_args is not None:
                span.set_attribute(semconv.ATTR_TOOL_ARGS_PREVIEW, tool_args[:preview_chars])
                span.set_attribute(semconv.ATTR_TOOL_ARGS_SHA256, _sha256_hex(tool_args))
            yield ToolCallContext(
                span=span,
                decision=env_ctx.decision,
                call_id=call_id,
            )

    @contextmanager
    def artifact(
        self,
        *,
        kind: str,
        artifact_id: Optional[str] = None,
        name: Optional[str] = None,
        path: Optional[str] = None,
        content: Optional[str] = None,
        size_bytes: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[ArtifactContext]:
        aid = artifact_id or f"artifact-{uuid.uuid4().hex[:12]}"
        sha: Optional[str] = None
        computed_size = size_bytes

        if content is not None:
            sha = _sha256_hex(content)
            computed_size = len(content.encode("utf-8"))
        elif path is not None and os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
            sha = hashlib.sha256(data).hexdigest()
            computed_size = len(data)

        span_name = f"{semconv.SPAN_ARTIFACT} {kind}"
        with self._tracer.start_as_current_span(span_name, kind=SpanKind.INTERNAL) as span:
            span.set_attribute(semconv.ATTR_ARTIFACT_ID, aid)
            span.set_attribute(semconv.ATTR_ARTIFACT_KIND, kind)
            _set_attr(span, semconv.ATTR_ARTIFACT_NAME, name)
            _set_attr(span, semconv.ATTR_ARTIFACT_PATH, path)
            _set_attr(span, semconv.ATTR_ARTIFACT_SHA256, sha)
            _set_attr(span, semconv.ATTR_ARTIFACT_SIZE_BYTES, computed_size)
            _set_metadata(span, metadata, "llmmas.artifact.meta")

            if message_store.is_enabled():
                message_store.write_artifact(
                    artifact_id=aid,
                    kind=kind,
                    name=name,
                    path=path,
                    sha256=sha,
                    size_bytes=computed_size,
                    metadata=dict(metadata or {}),
                )

            yield ArtifactContext(span=span, artifact_id=aid)

    @contextmanager
    def llm_call(
        self,
        *,
        provider_name: str,
        model: str,
        operation_name: str = semconv.GEN_AI_OPERATION_INFERENCE,
        request_id: Optional[str] = None,
        input_text: Optional[str] = None,
        preview_chars: int = 200,
        record_input: bool = True,
        agent_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Iterator[LLMCallContext]:
        seg = message_store.current_segment() or {}
        session_id = message_store.current_session_id()

        rid = request_id or f"llmreq-{uuid.uuid4().hex[:12]}"

        decision = None
        try:
            from .injection import DecisionKind, HookContext, HookType, get_engine, is_enabled
        except Exception:
            HookContext = None
            HookType = None
            get_engine = None
            is_enabled = lambda: False
            DecisionKind = None

        if is_enabled() and HookContext is not None:
            ctx = HookContext(
                hook_type=HookType.LLM_CALL,
                session_id=session_id,
                phase_name=seg.get("name"),
                phase_order=seg.get("order"),
                agent_id=agent_id,
                tool_name=None,
                extras={
                    "provider": provider_name,
                    "model": model,
                    "operation": operation_name,
                    "request_id": rid,
                },
            )
            decision = get_engine().decide(ctx, payload=input_text)

            if decision.kind == DecisionKind.DELAY and decision.delay_ms is not None:
                time.sleep(decision.delay_ms / 1000.0)

        token = _CURRENT_LLM_CALL_DECISION.set(decision)

        span_name = f"{operation_name} {model}"
        try:
            with self._tracer.start_as_current_span(span_name, kind=SpanKind.CLIENT) as span:
                span.set_attribute(semconv.ATTR_GEN_AI_OPERATION_NAME, operation_name)
                span.set_attribute(semconv.ATTR_GEN_AI_PROVIDER_NAME, provider_name)
                span.set_attribute(semconv.ATTR_GEN_AI_REQUEST_MODEL, model)
                span.set_attribute(semconv.ATTR_GEN_AI_REQUEST_ID, rid)
                _set_attr(span, semconv.ATTR_AGENT_ID, agent_id)
                _set_metadata(span, metadata, "llmmas.llm.meta")

                _annotate_fault_on_span(span, decision)

                if record_input and input_text is not None:
                    span.set_attribute(semconv.ATTR_LLM_INPUT_PREVIEW, input_text[:preview_chars])
                    span.set_attribute(semconv.ATTR_LLM_INPUT_SHA256, _sha256_hex(input_text))

                yield LLMCallContext(span=span, decision=decision, request_id=rid)
        finally:
            _CURRENT_LLM_CALL_DECISION.reset(token)


default_span_factory = SpanFactory()