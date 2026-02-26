from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .types import HookType


@dataclass(frozen=True)
class FaultSelector:
    """
    Matches where a fault applies. All fields are optional; absent fields are 'don't care'.
    Keys not recognized are stored in `extras`.
    """
    phase_name: Optional[str] = None
    phase_order: Optional[int] = None

    agent_id: Optional[str] = None
    step_index: Optional[int] = None

    source_agent_id: Optional[str] = None
    target_agent_id: Optional[str] = None
    edge_id: Optional[str] = None
    message_id: Optional[str] = None
    channel: Optional[str] = None

    tool_name: Optional[str] = None
    tool_type: Optional[str] = None
    tool_call_id: Optional[str] = None

    extras: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FaultSelector":
        known = {
            "phase_name", "phase_order",
            "agent_id", "step_index",
            "source_agent_id", "target_agent_id", "edge_id", "message_id", "channel",
            "tool_name", "tool_type", "tool_call_id",
        }
        kwargs = {k: d.get(k) for k in known if k in d}
        extras = {k: v for k, v in d.items() if k not in known}
        return FaultSelector(**kwargs, extras=extras)


@dataclass(frozen=True)
class FaultAction:
    """
    What to do when a selector matches.
    Examples:
      - type: "a2a.truncate" params: {"max_chars": 80}
      - type: "a2a.drop"
      - type: "tool.not_installed"
    """
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FaultAction":
        if "type" not in d or not isinstance(d["type"], str) or not d["type"].strip():
            raise ValueError("Fault action must have a non-empty string field 'type'")
        params = d.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("Fault action 'params' must be a dict if provided")
        return FaultAction(type=d["type"].strip(), params=params)


@dataclass(frozen=True)
class FaultLimits:
    """
    Simple activation limits. M2.2 will implement these.
    - probability: [0,1]
    - max_times: max number of applications (per session by default)
    """
    probability: float = 1.0
    max_times: Optional[int] = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FaultLimits":
        prob = d.get("probability", 1.0)
        if not isinstance(prob, (int, float)) or prob < 0.0 or prob > 1.0:
            raise ValueError("limits.probability must be a number in [0,1]")
        max_times = d.get("max_times", None)
        if max_times is not None and (not isinstance(max_times, int) or max_times <= 0):
            raise ValueError("limits.max_times must be a positive int if provided")
        return FaultLimits(probability=float(prob), max_times=max_times)


@dataclass(frozen=True)
class FaultSpec:
    """
    One fault specification entry loaded from YAML/JSON.
    """
    id: str
    hooks: list[HookType]
    selector: FaultSelector
    action: FaultAction
    limits: FaultLimits = field(default_factory=FaultLimits)
    description: Optional[str] = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "FaultSpec":
        # id
        fid = d.get("id")
        if not isinstance(fid, str) or not fid.strip():
            raise ValueError("Fault must have a non-empty string field 'id'")
        fid = fid.strip()

        # hooks: accept 'hook' or 'hooks'
        hooks_raw = d.get("hooks", d.get("hook"))
        if hooks_raw is None:
            raise ValueError(f"Fault '{fid}' must have 'hook' or 'hooks'")
        if isinstance(hooks_raw, str):
            hooks_list = [hooks_raw]
        elif isinstance(hooks_raw, list) and all(isinstance(x, str) for x in hooks_raw):
            hooks_list = hooks_raw
        else:
            raise ValueError(f"Fault '{fid}': 'hook(s)' must be a string or list of strings")

        hooks: list[HookType] = []
        for h in hooks_list:
            try:
                hooks.append(HookType(h))
            except Exception:
                raise ValueError(f"Fault '{fid}': unknown hook '{h}'. Allowed: {[e.value for e in HookType]}")

        # selector
        sel_raw = d.get("selector") or {}
        if not isinstance(sel_raw, dict):
            raise ValueError(f"Fault '{fid}': 'selector' must be a dict")
        selector = FaultSelector.from_dict(sel_raw)

        # action
        act_raw = d.get("action")
        if not isinstance(act_raw, dict):
            raise ValueError(f"Fault '{fid}': 'action' must be a dict")
        action = FaultAction.from_dict(act_raw)

        # limits (optional)
        lim_raw = d.get("limits") or {}
        if not isinstance(lim_raw, dict):
            raise ValueError(f"Fault '{fid}': 'limits' must be a dict if provided")
        limits = FaultLimits.from_dict(lim_raw)

        desc = d.get("description")
        if desc is not None and not isinstance(desc, str):
            raise ValueError(f"Fault '{fid}': 'description' must be a string if provided")

        return FaultSpec(id=fid, hooks=hooks, selector=selector, action=action, limits=limits, description=desc)