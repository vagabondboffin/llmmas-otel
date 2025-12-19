from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, Optional

from opentelemetry import trace

from . import semconv

_tracer = trace.get_tracer("llmmas-otel")


def observe_session(session_id: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _tracer.start_as_current_span(semconv.SPAN_SESSION) as span:
                span.set_attribute(semconv.ATTR_SESSION_ID, session_id)
                return fn(*args, **kwargs)
        return wrapper
    return deco


def observe_agent_step(*, agent_id: str, step_index: int) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _tracer.start_as_current_span(semconv.SPAN_AGENT_STEP) as span:
                span.set_attribute(semconv.ATTR_AGENT_ID, agent_id)
                span.set_attribute(semconv.ATTR_STEP_INDEX, step_index)
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
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with _tracer.start_as_current_span(semconv.SPAN_A2A_CLIENT) as span:
                span.set_attribute(semconv.ATTR_SOURCE_AGENT_ID, source_agent_id)
                span.set_attribute(semconv.ATTR_TARGET_AGENT_ID, target_agent_id)
                span.set_attribute(semconv.ATTR_EDGE_ID, edge_id)
                span.set_attribute(semconv.ATTR_MESSAGE_ID, message_id)
                if channel is not None:
                    span.set_attribute("llmmas.channel", channel)
                return fn(*args, **kwargs)
        return wrapper
    return deco
