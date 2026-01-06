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
    """Convenience context manager for segments/phases.

    NOTE: "order" is the proposal term. "index" is accepted as an alias for now.
    """
    if index is not None and order == 0:
        order = index
    return default_span_factory.segment(name=name, order=order, origin=origin)

def phase(*, name: str, order: int = 0, origin: Optional[str] = None, index: Optional[int] = None):
    """Alias for segment() using SE-friendly terminology."""
    return segment(name=name, order=order, origin=origin, index=index)


def observe_phase(*, name: str, order: int = 0, origin: Optional[str] = None, index: Optional[int] = None):
    """Alias for observe_segment() using SE-friendly terminology."""
    return observe_segment(name=name, order=order, origin=origin, index=index)


def observe_segment(*, name: str, order: int = 0, origin: Optional[str] = None, index: Optional[int] = None):
    """Decorator form for segments/phases."""
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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Observe an A2A *send* boundary.

    message_body_fn: extracts message body from (*args, **kwargs)
    carrier_fn: extracts a *mutable* mapping (e.g., headers dict) to inject trace context into
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            body: Optional[str] = message_body_fn(*args, **kwargs) if message_body_fn else None
            carrier: Optional[MutableMapping[str, str]] = carrier_fn(*args, **kwargs) if carrier_fn else None

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
            ):
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
    """Observe an A2A *receive/process* boundary.

    carrier_fn: extracts a mapping (e.g., headers dict) to extract trace context from.
    """

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
    """Observe a tool execution boundary."""

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tool_args: Optional[str] = tool_args_fn(*args, **kwargs) if tool_args_fn else None

            result: Any
            with default_span_factory.tool_call(
                tool_name=tool_name,
                tool_type=tool_type,
                tool_call_id=tool_call_id,
                tool_args=tool_args,
                tool_result=None,
                preview_chars=preview_chars,
                record_args=record_args,
                record_result=False,
            ) as _span:
                result = fn(*args, **kwargs)

                # If the user wants to record result preview/hash, do it after the call.
                if record_result and tool_result_fn is not None:
                    tool_result = tool_result_fn(result)
                    # We can't "re-enter" the span factory here, but we can set attributes directly.
                    if tool_result is not None:
                        from . import semconv
                        import hashlib

                        _span.set_attribute(semconv.ATTR_TOOL_RESULT_PREVIEW, tool_result[:preview_chars])
                        _span.set_attribute(
                            semconv.ATTR_TOOL_RESULT_SHA256,
                            hashlib.sha256(tool_result.encode("utf-8")).hexdigest(),
                        )

            return result

        return wrapper

    return deco
