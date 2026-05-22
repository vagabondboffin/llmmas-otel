from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator, Optional


# ---- Context for correlating offline records with session/workflow ----
_current_session_id: ContextVar[Optional[str]] = ContextVar("llmmas_session_id", default=None)
_current_workflow_stack: ContextVar[tuple[dict[str, Any], ...]] = ContextVar(
    "llmmas_workflow_stack",
    default=(),
)


@contextmanager
def session_context(session_id: str) -> Iterator[None]:
    token = _current_session_id.set(session_id)
    try:
        yield
    finally:
        _current_session_id.reset(token)


@contextmanager
def workflow_context(
    *,
    workflow_id: str,
    name: str,
    order: int = 0,
    kind: str = "workflow",
    origin: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> Iterator[None]:
    stack = _current_workflow_stack.get()
    if parent_id is None and stack:
        parent_id = stack[-1].get("id")

    record: dict[str, Any] = {
        "id": workflow_id,
        "name": name,
        "order": order,
        "kind": kind,
        "origin": origin,
        "parent_id": parent_id,
        "depth": len(stack),
    }
    token = _current_workflow_stack.set((*stack, record))
    try:
        yield
    finally:
        _current_workflow_stack.reset(token)


@contextmanager
def segment_context(name: str, order: int, origin: Optional[str] = None) -> Iterator[None]:
    """Backward-compatible alias for workflow_context(kind='segment')."""
    workflow_id = f"segment:{order}:{name}"
    with workflow_context(
        workflow_id=workflow_id,
        name=name,
        order=order,
        kind="segment",
        origin=origin,
    ):
        yield


def current_session_id() -> Optional[str]:
    return _current_session_id.get()


def current_workflow() -> Optional[dict[str, Any]]:
    stack = _current_workflow_stack.get()
    if not stack:
        return None
    return dict(stack[-1])


def current_workflow_stack() -> list[dict[str, Any]]:
    return [dict(item) for item in _current_workflow_stack.get()]


def current_segment() -> Optional[dict[str, Any]]:
    """Backward-compatible view used by older fault-injection code."""
    workflow = current_workflow()
    if workflow is None:
        return None
    return {
        "id": workflow.get("id"),
        "name": workflow.get("name"),
        "order": workflow.get("order"),
        "kind": workflow.get("kind"),
        "origin": workflow.get("origin"),
        "parent_id": workflow.get("parent_id"),
        "depth": workflow.get("depth"),
    }


# ---- Message store configuration ----
@dataclass(frozen=True)
class MessageStoreConfig:
    path: str


_config: Optional[MessageStoreConfig] = None


def enable_message_store(path: str) -> None:
    """
    Enable JSONL message storage for offline analysis.

    The trace keeps previews and hashes. This store keeps full message bodies and
    selected metadata. Safe default: disabled until explicitly enabled.
    """
    global _config
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _config = MessageStoreConfig(path=path)


def disable_message_store() -> None:
    global _config
    _config = None


def is_enabled() -> bool:
    return _config is not None


def _append_jsonl(record: dict[str, Any]) -> None:
    if _config is None:
        return
    with open(_config.path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_message(
    *,
    direction: str,
    message_id: str,
    sha256: str,
    body: str,
    source_agent_id: str,
    target_agent_id: str,
    edge_id: str,
    channel: Optional[str] = None,
    message_kind: Optional[str] = None,
    route_via: Optional[str] = None,
    parent_message_id: Optional[str] = None,
    original_sha256: Optional[str] = None,
    fault_spec_id: Optional[str] = None,
    fault_type: Optional[str] = None,
    fault_decision: Optional[str] = None,
    dropped: bool = False,
) -> None:
    """Append an A2A/routed-message record to the JSONL store, if enabled."""
    if _config is None:
        return

    record: dict[str, Any] = {
        "record_type": "message",
        "session_id": current_session_id(),
        "workflow": current_workflow(),
        "workflow_stack": current_workflow_stack(),
        "direction": direction,
        "message_id": message_id,
        "parent_message_id": parent_message_id,
        "sha256": sha256,
        "source_agent_id": source_agent_id,
        "target_agent_id": target_agent_id,
        "edge_id": edge_id,
        "channel": channel,
        "message_kind": message_kind,
        "route_via": route_via,
        "body": body,
        "dropped": dropped,
    }

    if original_sha256 is not None:
        record["original_sha256"] = original_sha256
    if fault_spec_id is not None:
        record["fault_spec_id"] = fault_spec_id
    if fault_type is not None:
        record["fault_type"] = fault_type
    if fault_decision is not None:
        record["fault_decision"] = fault_decision

    _append_jsonl(record)


def write_artifact(
    *,
    artifact_id: str,
    kind: str,
    name: Optional[str] = None,
    path: Optional[str] = None,
    sha256: Optional[str] = None,
    size_bytes: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Append an artifact/state-delta record to the JSONL store, if enabled."""
    if _config is None:
        return

    record: dict[str, Any] = {
        "record_type": "artifact",
        "session_id": current_session_id(),
        "workflow": current_workflow(),
        "workflow_stack": current_workflow_stack(),
        "artifact_id": artifact_id,
        "kind": kind,
        "name": name,
        "path": path,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "metadata": metadata or {},
    }
    _append_jsonl(record)