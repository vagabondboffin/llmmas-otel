from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class HookType(str, Enum):
    A2A_SEND = "a2a_send"
    A2A_RECEIVE = "a2a_receive"
    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"  # placeholder for later


@dataclass(frozen=True)
class HookContext:
    """
    Framework-agnostic context passed to the fault injection engine.
    Most fields are optional because not every hook has all fields.
    """
    hook_type: HookType

    # Run/phase context
    session_id: Optional[str] = None
    phase_name: Optional[str] = None
    phase_order: Optional[int] = None

    # Agent-step context
    agent_id: Optional[str] = None
    step_index: Optional[int] = None

    # A2A context
    source_agent_id: Optional[str] = None
    target_agent_id: Optional[str] = None
    edge_id: Optional[str] = None
    message_id: Optional[str] = None
    channel: Optional[str] = None

    # Tool context
    tool_name: Optional[str] = None
    tool_type: Optional[str] = None
    tool_call_id: Optional[str] = None

    # Free-form extras (future-proof)
    extras: dict[str, Any] = field(default_factory=dict)


class DecisionKind(str, Enum):
    PASS = "pass"
    DROP = "drop"
    DELAY = "delay"
    MUTATE = "mutate"
    RAISE = "raise"
    RETURN = "return"


@dataclass(frozen=True)
class InjectionDecision:
    """
    Output of the fault injection engine.
    """
    kind: DecisionKind = DecisionKind.PASS

    # Identifiers for trace attribution
    fault_id: Optional[str] = None          # stable ID from config, e.g. "F01"
    fault_type: Optional[str] = None        # e.g. "a2a.truncate", "tool.not_installed"

    # Action parameters
    delay_ms: Optional[int] = None
    mutated_payload: Optional[str] = None   # for MUTATE (message body or tool args/result)
    raise_exception: Optional[Exception] = None
    return_value: Any = None                # for RETURN

    # Extra metadata to record on spans / message store
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def pass_through() -> "InjectionDecision":
        return InjectionDecision(kind=DecisionKind.PASS)

    @staticmethod
    def drop(*, fault_id: str, fault_type: str, metadata: Optional[dict[str, Any]] = None) -> "InjectionDecision":
        return InjectionDecision(kind=DecisionKind.DROP, fault_id=fault_id, fault_type=fault_type, metadata=metadata or {})

    @staticmethod
    def delay(*, fault_id: str, fault_type: str, delay_ms: int, metadata: Optional[dict[str, Any]] = None) -> "InjectionDecision":
        return InjectionDecision(kind=DecisionKind.DELAY, fault_id=fault_id, fault_type=fault_type, delay_ms=delay_ms, metadata=metadata or {})

    @staticmethod
    def mutate(*, fault_id: str, fault_type: str, mutated_payload: str, metadata: Optional[dict[str, Any]] = None) -> "InjectionDecision":
        return InjectionDecision(kind=DecisionKind.MUTATE, fault_id=fault_id, fault_type=fault_type, mutated_payload=mutated_payload, metadata=metadata or {})

    @staticmethod
    def raise_(*, fault_id: str, fault_type: str, exc: Exception, metadata: Optional[dict[str, Any]] = None) -> "InjectionDecision":
        return InjectionDecision(kind=DecisionKind.RAISE, fault_id=fault_id, fault_type=fault_type, raise_exception=exc, metadata=metadata or {})

    @staticmethod
    def return_(*, fault_id: str, fault_type: str, value: Any, metadata: Optional[dict[str, Any]] = None) -> "InjectionDecision":
        return InjectionDecision(kind=DecisionKind.RETURN, fault_id=fault_id, fault_type=fault_type, return_value=value, metadata=metadata or {})