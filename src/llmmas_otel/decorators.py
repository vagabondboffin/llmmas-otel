from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, Optional, MutableMapping, Mapping

from .span_factory import default_span_factory


def observe_session(session_id: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with default_span_factory.session(session_id=session_id):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def segment(*, name: str, order: int = 0, origin: Optional[str] = None, index: Optional[int] = None):
    if index is not None and order == 0:
        order = index
    return default_span_factory.segment(name=name, order=order, origin=origin)


def phase(*, name: str, order: int = 0, origin: Optional[str] = None, index: Optional[int] = None):
    return segment(name=name, order=order, origin=origin, index=index)


def observe_phase(*, name: str, order: int = 0, origin: Optional[str] = None, index: Optional[int] = None):
    return observe_segment(name=name, order=order, origin=origin, index=index)


def observe_segment(*, name: str, order: int = 0, origin: Optional[str] = None, index: Optional[int] = None):
    if index is not None and order == 0:
        order = index

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with default_span_factory.segment(name=name, order=order, origin=origin):
                return fn(*args, **kwargs)
        return wrapper
    return deco


def observe_agent_step(*, agent_id: str, step_index: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with default_span_factory.agent_step(agent_id=agent_id, step_index=step_index):
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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            body: Optional[str] = message_body_fn(*args, **kwargs) if message_body_fn else None
            carrier: Optional[MutableMapping[str, str]] = carrier_fn(*args, **kwargs) if carrier_fn else None

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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            body: Optional[str] = message_body_fn(*args, **kwargs) if message_body_fn else None
            carrier: Optional[Mapping[str, str]] = carrier_fn(*args, **kwargs) if carrier_fn else None

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
            tool_args: Optional[str] = tool_args_fn(*args, **kwargs) if tool_args_fn else None

            with default_span_factory.tool_call(
                tool_name=tool_name,
                tool_type=tool_type,
                tool_call_id=tool_call_id,
                tool_args=tool_args,
                preview_chars=preview_chars,
                record_args=record_args,
            ) as ctx:
                dec = default_span_factory.current_tool_call_decision()

                try:
                    from .injection import DecisionKind
                except Exception:
                    DecisionKind = None  # type: ignore

                if dec is not None and DecisionKind is not None:
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
                        result = getattr(dec, "return_value", None)
                    else:
                        result = fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)

                if record_result and tool_result_fn is not None:
                    try:
                        tool_result = tool_result_fn(result)
                    except Exception:
                        tool_result = None
                    if tool_result is not None:
                        from . import semconv
                        ctx.span.set_attribute(semconv.ATTR_TOOL_RESULT_PREVIEW, tool_result[:preview_chars])
                        ctx.span.set_attribute(semconv.ATTR_TOOL_RESULT_SHA256, __import__("hashlib").sha256(tool_result.encode("utf-8")).hexdigest())

                return result
        return wrapper
    return deco


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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Observe an LLM call boundary with fault injection support.

    - Span created by SpanFactory.llm_call(...)
    - PASS/DELAY: calls the underlying function
    - RETURN: returns injected return_value (skips call)
    - RAISE: raises injected exception (skips call) and records exception on span
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            input_text: Optional[str] = input_text_fn(*args, **kwargs) if input_text_fn else None

            with default_span_factory.llm_call(
                provider_name=provider_name,
                model=model,
                operation_name=operation_name,
                request_id=request_id,
                input_text=input_text,
                preview_chars=preview_chars,
                record_input=record_input,
            ) as ctx:
                dec = default_span_factory.current_llm_call_decision()

                try:
                    from .injection import DecisionKind
                except Exception:
                    DecisionKind = None  # type: ignore

                if dec is not None and DecisionKind is not None:
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
                        ctx.span.set_attribute(semconv.ATTR_LLM_OUTPUT_SHA256, __import__("hashlib").sha256(out_text.encode("utf-8")).hexdigest())

                return result
        return wrapper
    return deco