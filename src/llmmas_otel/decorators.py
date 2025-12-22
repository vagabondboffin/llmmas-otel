from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any, Optional

from .span_factory import default_span_factory


def observe_session(session_id: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with default_span_factory.session(session_id=session_id):
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
    preview_chars: int = 200,
    add_event: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Observe an A2A send boundary.

    message_body_fn: function that extracts the message body from (*args, **kwargs)
    so users can adapt this decorator to whatever their framework uses.
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            body: Optional[str] = None
            if message_body_fn is not None:
                body = message_body_fn(*args, **kwargs)

            with default_span_factory.a2a_send(
                source_agent_id=source_agent_id,
                target_agent_id=target_agent_id,
                edge_id=edge_id,
                message_id=message_id,
                channel=channel,
                message_body=body,
                preview_chars=preview_chars,
                add_event=add_event,
            ):
                return fn(*args, **kwargs)
        return wrapper
    return deco