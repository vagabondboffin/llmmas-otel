from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, Mapping, MutableMapping, Optional

from .span_factory import default_span_factory


def observe_session(
    session_id: str,
    *,
    name: Optional[str] = None,
    task_id: Optional[str] = None,
    framework: Optional[str] = None,
    system: Optional[str] = None,
    adapter: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with default_span_factory.session(
                session_id=session_id,
                name=name,
                task_id=task_id,
                framework=framework,
                system=system,
                adapter=adapter,
                metadata=metadata,
            ):
                return fn(*args, **kwargs)

        return wrapper

    return deco


def workflow(
    *,
    name: str,
    order: int = 0,
    kind: str = "workflow",
    origin: Optional[str] = None,
    workflow_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    index: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    if index is not None and order == 0:
        order = index
    return default_span_factory.workflow(
        name=name,
        order=order,
        kind=kind,
        origin=origin,
        workflow_id=workflow_id,
        parent_id=parent_id,
        metadata=metadata,
    )


def segment(
    *,
    name: str,
    order: int = 0,
    origin: Optional[str] = None,
    index: Optional[int] = None,
    workflow_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    kind: str = "segment",
    metadata: Optional[Mapping[str, Any]] = None,
):
    if index is not None and order == 0:
        order = index
    return default_span_factory.segment(
        name=name,
        order=order,
        origin=origin,
        workflow_id=workflow_id,
        parent_id=parent_id,
        kind=kind,
        metadata=metadata,
    )


def phase(
    *,
    name: str,
    order: int = 0,
    origin: Optional[str] = None,
    index: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    return segment(
        name=name,
        order=order,
        origin=origin,
        index=index,
        kind="phase",
        metadata=metadata,
    )


def observe_workflow(
    *,
    name: str,
    order: int = 0,
    kind: str = "workflow",
    origin: Optional[str] = None,
    workflow_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    index: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    if index is not None and order == 0:
        order = index

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with default_span_factory.workflow(
                name=name,
                order=order,
                kind=kind,
                origin=origin,
                workflow_id=workflow_id,
                parent_id=parent_id,
                metadata=metadata,
            ):
                return fn(*args, **kwargs)

        return wrapper

    return deco


def observe_phase(
    *,
    name: str,
    order: int = 0,
    origin: Optional[str] = None,
    index: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    return observe_segment(
        name=name,
        order=order,
        origin=origin,
        index=index,
        kind="phase",
        metadata=metadata,
    )


def observe_segment(
    *,
    name: str,
    order: int = 0,
    origin: Optional[str] = None,
    index: Optional[int] = None,
    workflow_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    kind: str = "segment",
    metadata: Optional[Mapping[str, Any]] = None,
):
    if index is not None and order == 0:
        order = index

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with default_span_factory.segment(
                name=name,
                order=order,
                origin=origin,
                workflow_id=workflow_id,
                parent_id=parent_id,
                kind=kind,
                metadata=metadata,
            ):
                return fn(*args, **kwargs)

        return wrapper

    return deco


def observe_agent_step(
    *,
    agent_id: str,
    step_index: int,
    agent_role: Optional[str] = None,
    agent_impl: Optional[str] = None,
    parent_agent_id: Optional[str] = None,
    step_kind: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with default_span_factory.agent_step(
                agent_id=agent_id,
                step_index=step_index,
                agent_role=agent_role,
                agent_impl=agent_impl,
                parent_agent_id=parent_agent_id,
                step_kind=step_kind,
                metadata=metadata,
            ):
                return fn(*args, **kwargs)

        return wrapper

    return deco


def observe_delegation(
    *,
    from_agent_id: str,
    to_agent_id: str,
    delegation_id: Optional[str] = None,
    task_id: Optional[str] = None,
    kind: str = "delegation",
    via: Optional[str] = None,
    goal_fn: Optional[Callable[..., Optional[str]]] = None,
    preview_chars: int = 200,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            goal = goal_fn(*args, **kwargs) if goal_fn else None
            with default_span_factory.delegation(
                from_agent_id=from_agent_id,
                to_agent_id=to_agent_id,
                delegation_id=delegation_id,
                task_id=task_id,
                kind=kind,
                via=via,
                goal=goal,
                preview_chars=preview_chars,
                metadata=metadata,
            ):
                return fn(*args, **kwargs)

        return wrapper

    return deco


def observe_a2a_send(
    *,
    source_agent_id: str,
    target_agent_id: str,
    edge_id: str,
    message_id: str,
    channel: Optional[str] = None,
    message_body_fn: Optional[Callable[..., Optional[str]]] = None,
    carrier_fn: Optional[Callable[..., Optional[MutableMapping[str, str]]]] = None,
    propagate_context: bool = True,
    preview_chars: int = 200,
    add_event: bool = True,
    message_body_setter_fn: Optional[Callable[..., None]] = None,
    route_via: Optional[str] = None,
    message_kind: Optional[str] = None,
    parent_message_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            body = message_body_fn(*args, **kwargs) if message_body_fn else None
            carrier = carrier_fn(*args, **kwargs) if carrier_fn else None

            apply_mutation = None
            if message_body_setter_fn is not None:
                apply_mutation = lambda new_body: message_body_setter_fn(new_body, *args, **kwargs)

            with default_span_factory.a2a_send(
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                edge_id=edge_id,
                message_id=message_id,
                channel=channel,
                message_body=body,
                carrier=carrier,
                propagate_context=propagate_context,
                preview_chars=preview_chars,
                add_event=add_event,
                apply_mutation=apply_mutation,
                route_via=route_via,
                message_kind=message_kind,
                parent_message_id=parent_message_id,
                metadata=metadata,
            ) as ctx:
                try:
                    from .injection import DecisionKind

                    if ctx.decision is not None and ctx.decision.kind == DecisionKind.DROP:
                        return None
                except Exception:
                    pass
                return fn(*args, **kwargs)

        return wrapper

    return deco


def observe_a2a_receive(
    *,
    source_agent_id: str,
    target_agent_id: str,
    edge_id: str,
    message_id: str,
    channel: Optional[str] = None,
    message_body_fn: Optional[Callable[..., Optional[str]]] = None,
    carrier_fn: Optional[Callable[..., Optional[Mapping[str, str]]]] = None,
    link_from_carrier: bool = True,
    preview_chars: int = 200,
    add_event: bool = True,
    route_via: Optional[str] = None,
    message_kind: Optional[str] = None,
    parent_message_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            body = message_body_fn(*args, **kwargs) if message_body_fn else None
            carrier = carrier_fn(*args, **kwargs) if carrier_fn else None

            with default_span_factory.a2a_receive(
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                edge_id=edge_id,
                message_id=message_id,
                channel=channel,
                message_body=body,
                carrier=carrier,
                link_from_carrier=link_from_carrier,
                preview_chars=preview_chars,
                add_event=add_event,
                route_via=route_via,
                message_kind=message_kind,
                parent_message_id=parent_message_id,
                metadata=metadata,
            ):
                try:
                    from .injection import DecisionKind

                    dec = default_span_factory.current_a2a_receive_decision()
                    if dec is not None and getattr(dec, "kind", None) == DecisionKind.DROP:
                        return None
                except Exception:
                    pass
                return fn(*args, **kwargs)

        return wrapper

    return deco


def observe_environment_action(
    *,
    name: str,
    kind: str = "tool",
    action_id: Optional[str] = None,
    input_text_fn: Optional[Callable[..., Optional[str]]] = None,
    output_text_fn: Optional[Callable[[Any], Optional[str]]] = None,
    preview_chars: int = 200,
    record_input: bool = False,
    record_output: bool = False,
    changed_files_fn: Optional[Callable[[Any], Optional[list[str]]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_text = input_text_fn(*args, **kwargs) if input_text_fn else None

            with default_span_factory.environment_action(
                name=name,
                kind=kind,
                action_id=action_id,
                input_text=input_text,
                preview_chars=preview_chars,
                record_input=record_input,
                metadata=metadata,
            ) as ctx:
                result = _run_with_tool_fault_decision(fn, ctx, *args, **kwargs)

                if record_output and output_text_fn is not None:
                    try:
                        output_text = output_text_fn(result)
                    except Exception:
                        output_text = None
                    if output_text is not None:
                        from . import semconv

                        ctx.span.set_attribute(
                            semconv.ATTR_ENV_ACTION_OUTPUT_PREVIEW,
                            output_text[:preview_chars],
                        )
                        ctx.span.set_attribute(
                            semconv.ATTR_ENV_ACTION_OUTPUT_SHA256,
                            __import__("hashlib").sha256(output_text.encode("utf-8")).hexdigest(),
                        )

                if changed_files_fn is not None:
                    try:
                        changed_files = changed_files_fn(result)
                    except Exception:
                        changed_files = None
                    if changed_files:
                        from . import semconv

                        ctx.span.set_attribute(semconv.ATTR_ENV_ACTION_CHANGED_FILES, changed_files)

                return result

        return wrapper

    return deco


def observe_tool_call(
    *,
    tool_name: str,
    tool_type: Optional[str] = None,
    tool_call_id: Optional[str] = None,
    tool_args_fn: Optional[Callable[..., Optional[str]]] = None,
    tool_result_fn: Optional[Callable[..., Optional[str]]] = None,
    preview_chars: int = 200,
    record_args: bool = False,
    record_result: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tool_args = tool_args_fn(*args, **kwargs) if tool_args_fn else None

            with default_span_factory.tool_call(
                tool_name=tool_name,
                tool_type=tool_type,
                tool_call_id=tool_call_id,
                tool_args=tool_args,
                preview_chars=preview_chars,
                record_args=record_args,
            ) as ctx:
                result = _run_with_tool_fault_decision(fn, ctx, *args, **kwargs)

                if record_result and tool_result_fn is not None:
                    try:
                        tool_result = tool_result_fn(result)
                    except Exception:
                        tool_result = None
                    if tool_result is not None:
                        from . import semconv

                        sha = __import__("hashlib").sha256(tool_result.encode("utf-8")).hexdigest()
                        ctx.span.set_attribute(semconv.ATTR_TOOL_RESULT_PREVIEW, tool_result[:preview_chars])
                        ctx.span.set_attribute(semconv.ATTR_TOOL_RESULT_SHA256, sha)
                        ctx.span.set_attribute(semconv.ATTR_ENV_ACTION_OUTPUT_PREVIEW, tool_result[:preview_chars])
                        ctx.span.set_attribute(semconv.ATTR_ENV_ACTION_OUTPUT_SHA256, sha)

                return result

        return wrapper

    return deco


def observe_artifact(
    *,
    kind: str,
    artifact_id: Optional[str] = None,
    name: Optional[str] = None,
    path_fn: Optional[Callable[..., Optional[str]]] = None,
    content_fn: Optional[Callable[[Any], Optional[str]]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = fn(*args, **kwargs)
            path = path_fn(*args, **kwargs) if path_fn else None
            content = content_fn(result) if content_fn else None
            with default_span_factory.artifact(
                kind=kind,
                artifact_id=artifact_id,
                name=name,
                path=path,
                content=content,
                metadata=metadata,
            ):
                pass
            return result

        return wrapper

    return deco


def artifact(
    *,
    kind: str,
    artifact_id: Optional[str] = None,
    name: Optional[str] = None,
    path: Optional[str] = None,
    content: Optional[str] = None,
    size_bytes: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
):
    return default_span_factory.artifact(
        kind=kind,
        artifact_id=artifact_id,
        name=name,
        path=path,
        content=content,
        size_bytes=size_bytes,
        metadata=metadata,
    )


def environment_action(
    *,
    name: str,
    kind: str = "tool",
    action_id: Optional[str] = None,
    input_text: Optional[str] = None,
    preview_chars: int = 200,
    record_input: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
):
    return default_span_factory.environment_action(
        name=name,
        kind=kind,
        action_id=action_id,
        input_text=input_text,
        preview_chars=preview_chars,
        record_input=record_input,
        metadata=metadata,
    )


def delegation(
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
):
    return default_span_factory.delegation(
        from_agent_id=from_agent_id,
        to_agent_id=to_agent_id,
        delegation_id=delegation_id,
        task_id=task_id,
        kind=kind,
        via=via,
        goal=goal,
        preview_chars=preview_chars,
        metadata=metadata,
    )


def observe_llm_call(
    *,
    provider_name: str,
    model: str,
    operation_name: str = "inference",
    request_id: Optional[str] = None,
    input_text_fn: Optional[Callable[..., Optional[str]]] = None,
    preview_chars: int = 200,
    record_input: bool = True,
    output_text_fn: Optional[Callable[[Any], Optional[str]]] = None,
    record_output: bool = False,
    agent_id: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_text = input_text_fn(*args, **kwargs) if input_text_fn else None

            with default_span_factory.llm_call(
                provider_name=provider_name,
                model=model,
                operation_name=operation_name,
                request_id=request_id,
                input_text=input_text,
                preview_chars=preview_chars,
                record_input=record_input,
                agent_id=agent_id,
                metadata=metadata,
            ) as ctx:
                dec = default_span_factory.current_llm_call_decision()

                try:
                    from .injection import DecisionKind
                except Exception:
                    DecisionKind = None  # type: ignore

                if dec is not None and DecisionKind is not None:
                    if dec.kind == DecisionKind.MUTATE_INPUT:
                        mutator = dec.return_value
                        try:
                            args, kwargs = mutator(args, kwargs)
                        except Exception as e:
                            ctx.span.set_attribute("llmmas.fault.error", f"mutator_failed: {e}")

                    if dec.kind == DecisionKind.RAISE:
                        exc = getattr(dec, "raise_exception", None) or RuntimeError("Injected LLM error")
                        try:
                            from opentelemetry.trace.status import Status, StatusCode

                            ctx.span.record_exception(exc)
                            ctx.span.set_status(Status(StatusCode.ERROR, str(exc)))
                        except Exception:
                            pass
                        raise exc

                    if dec.kind == DecisionKind.RETURN:
                        result = getattr(dec, "return_value", None)
                    else:
                        result = fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)

                if record_output and output_text_fn is not None:
                    try:
                        out_text = output_text_fn(result)
                    except Exception:
                        out_text = None
                    if out_text is not None:
                        from . import semconv

                        ctx.span.set_attribute(semconv.ATTR_LLM_OUTPUT_PREVIEW, out_text[:preview_chars])
                        ctx.span.set_attribute(
                            semconv.ATTR_LLM_OUTPUT_SHA256,
                            __import__("hashlib").sha256(out_text.encode("utf-8")).hexdigest(),
                        )

                return result

        return wrapper

    return deco


def _run_with_tool_fault_decision(fn: Callable[..., Any], ctx: Any, *args: Any, **kwargs: Any) -> Any:
    dec = default_span_factory.current_tool_call_decision()

    try:
        from .injection import DecisionKind
    except Exception:
        DecisionKind = None  # type: ignore

    if dec is not None and DecisionKind is not None:
        if dec.kind == DecisionKind.MUTATE_INPUT:
            args, kwargs = dec.return_value(args, kwargs)

        if dec.kind == DecisionKind.RAISE:
            exc = getattr(dec, "raise_exception", None) or RuntimeError("Injected tool error")
            try:
                from opentelemetry.trace.status import Status, StatusCode

                ctx.span.record_exception(exc)
                ctx.span.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:
                pass
            raise exc

        if dec.kind == DecisionKind.RETURN:
            return getattr(dec, "return_value", None)

    return fn(*args, **kwargs)