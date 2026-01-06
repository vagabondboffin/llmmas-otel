from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional


# ---- Context for correlating messages with session/segment ----
_current_session_id: ContextVar[Optional[str]] = ContextVar("llmmas_session_id", default=None)
_current_segment_name: ContextVar[Optional[str]] = ContextVar("llmmas_segment_name", default=None)
_current_segment_order: ContextVar[Optional[int]] = ContextVar("llmmas_segment_order", default=None)


@contextmanager
def session_context(session_id: str) -> Iterator[None]:
    token = _current_session_id.set(session_id)
    try:
        yield
    finally:
        _current_session_id.reset(token)


@contextmanager
def segment_context(name: str, order: int) -> Iterator[None]:
    token_name = _current_segment_name.set(name)
    token_order = _current_segment_order.set(order)
    try:
        yield
    finally:
        _current_segment_name.reset(token_name)
        _current_segment_order.reset(token_order)


def current_session_id() -> Optional[str]:
    return _current_session_id.get()


def current_segment() -> Optional[dict]:
    name = _current_segment_name.get()
    order = _current_segment_order.get()
    if name is None or order is None:
        return None
    return {"name": name, "order": order}


# ---- Message store configuration ----
@dataclass(frozen=True)
class MessageStoreConfig:
    path: str


_config: Optional[MessageStoreConfig] = None


def enable_message_store(path: str) -> None:
    """
    Enable JSONL message storage (full message bodies) for offline analysis.
    One JSON object per line. Safe default: disabled until explicitly enabled.

    Example:
      enable_message_store("out/messages.jsonl")
    """
    global _config
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _config = MessageStoreConfig(path=path)


def disable_message_store() -> None:
    global _config
    _config = None


def is_enabled() -> bool:
    return _config is not None


def write_message(
    *,
    direction: str,  # "send" | "receive"
    message_id: str,
    sha256: str,
    body: str,
    source_agent_id: str,
    target_agent_id: str,
    edge_id: str,
    channel: Optional[str] = None,
) -> None:
    """
    Append a message record to JSONL store, if enabled.
    """
    if _config is None:
        return

    record = {
        "session_id": current_session_id(),
        "segment": current_segment(),
        "direction": direction,
        "message_id": message_id,
        "sha256": sha256,
        "source_agent_id": source_agent_id,
        "target_agent_id": target_agent_id,
        "edge_id": edge_id,
        "channel": channel,
        "body": body,
    }

    with open(_config.path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
