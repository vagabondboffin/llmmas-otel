from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Any

from .engine import FaultEngine
from .matcher import selector_matches
from .spec import FaultSpec
from .types import HookContext, InjectionDecision, DecisionKind
from .exceptions import LLMRateLimitError, LLMNetworkError, LLMTimeoutError


def _stable_coin_flip(probability: float, *, seed: str, session_id: str, fault_id: str, attempt: int) -> bool:
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False

    raw = f"{seed}|{session_id}|{fault_id}|{attempt}".encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    v = int.from_bytes(digest[:8], "big") / 2**64
    return v < probability


@dataclass
class SpecFaultEngine(FaultEngine):
    """
    Spec-driven engine.
    Supports hooks:
      - A2A_SEND / A2A_RECEIVE
      - TOOL_CALL
      - LLM_CALL (M2.8/M3.0)
    """
    specs: list[FaultSpec]
    seed: str = "0"
    _counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def _session_key(self, ctx: HookContext) -> str:
        return ctx.session_id or "__global__"

    def _get_count(self, session_id: str, fault_id: str) -> int:
        return self._counts.get((session_id, fault_id), 0)

    def _inc_count(self, session_id: str, fault_id: str) -> int:
        key = (session_id, fault_id)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def decide(self, ctx: HookContext, payload: Optional[str] = None) -> InjectionDecision:
        session_id = self._session_key(ctx)

        for spec in self.specs:
            if ctx.hook_type not in spec.hooks:
                continue
            if not selector_matches(spec.selector, ctx):
                continue

            already = self._get_count(session_id, spec.id)
            if spec.limits.max_times is not None and already >= spec.limits.max_times:
                continue

            attempt = already + 1
            if not _stable_coin_flip(
                spec.limits.probability,
                seed=self.seed,
                session_id=session_id,
                fault_id=spec.id,
                attempt=attempt,
            ):
                continue

            decision = self._action_to_decision(spec, ctx, payload)

            if decision.kind != DecisionKind.PASS:
                self._inc_count(session_id, spec.id)

            return decision

        return InjectionDecision.pass_through()

    def _action_to_decision(self, spec: FaultSpec, ctx: HookContext, payload: Optional[str]) -> InjectionDecision:
        t = spec.action.type

        # ---------------- A2A faults ----------------
        if t == "a2a.drop":
            return InjectionDecision.drop(fault_id=spec.id, fault_type=t)

        if t == "a2a.delay":
            ms = spec.action.params.get("delay_ms", spec.action.params.get("ms"))
            if not isinstance(ms, int) or ms < 0:
                raise ValueError(f"Fault '{spec.id}': a2a.delay requires integer params.delay_ms (or ms) >= 0")
            return InjectionDecision.delay(fault_id=spec.id, fault_type=t, delay_ms=ms)

        if t == "a2a.truncate":
            if payload is None:
                raise ValueError(f"Fault '{spec.id}': a2a.truncate requires a string payload (message_body)")
            max_chars = spec.action.params.get("max_chars")
            if not isinstance(max_chars, int) or max_chars < 0:
                raise ValueError(f"Fault '{spec.id}': a2a.truncate requires integer params.max_chars >= 0")
            mutated = payload[:max_chars]
            meta = {"original_len": len(payload), "new_len": len(mutated), "max_chars": max_chars}
            return InjectionDecision.mutate(fault_id=spec.id, fault_type=t, mutated_payload=mutated, metadata=meta)

        # ---------------- TOOL faults ----------------
        if t == "tool.delay":
            ms = spec.action.params.get("delay_ms", spec.action.params.get("ms"))
            if not isinstance(ms, int) or ms < 0:
                raise ValueError(f"Fault '{spec.id}': tool.delay requires integer params.delay_ms (or ms) >= 0")
            return InjectionDecision.delay(fault_id=spec.id, fault_type=t, delay_ms=ms)

        if t == "tool.not_installed":
            tool = ctx.tool_name or "unknown_tool"
            exc = FileNotFoundError(f"Tool not installed: {tool}")
            return InjectionDecision.raise_(fault_id=spec.id, fault_type=t, exc=exc, metadata={"tool_name": tool})

        if t == "tool.timeout":
            tool = ctx.tool_name or "unknown_tool"
            exc = TimeoutError(f"Tool timeout: {tool}")
            return InjectionDecision.raise_(fault_id=spec.id, fault_type=t, exc=exc, metadata={"tool_name": tool})

        if t == "tool.malformed_response":
            val: Any = spec.action.params.get("return_value", spec.action.params.get("value", "MALFORMED_RESPONSE"))
            meta = {"returned_type": type(val).__name__}
            return InjectionDecision.return_(fault_id=spec.id, fault_type=t, value=val, metadata=meta)

        # ---------------- LLM faults (M2.8/M3.0) ----------------
        if t == "llm.delay":
            ms = spec.action.params.get("delay_ms", spec.action.params.get("ms"))
            if not isinstance(ms, int) or ms < 0:
                raise ValueError(f"Fault '{spec.id}': llm.delay requires integer params.delay_ms (or ms) >= 0")
            return InjectionDecision.delay(fault_id=spec.id, fault_type=t, delay_ms=ms)

        if t == "llm.rate_limit":
            exc = LLMRateLimitError("Injected LLM rate limit")
            return InjectionDecision.raise_(fault_id=spec.id, fault_type=t, exc=exc)

        if t == "llm.timeout":
            exc = LLMTimeoutError("Injected LLM timeout")
            return InjectionDecision.raise_(fault_id=spec.id, fault_type=t, exc=exc)

        if t == "llm.network_error":
            exc = LLMNetworkError("Injected LLM network error")
            return InjectionDecision.raise_(fault_id=spec.id, fault_type=t, exc=exc)

        if t == "llm.malformed_response":
            # arbitrary return allowed; default is clearly malformed
            val: Any = spec.action.params.get("return_value", spec.action.params.get("value", "MALFORMED_LLM_RESPONSE"))
            meta = {"returned_type": type(val).__name__}
            return InjectionDecision.return_(fault_id=spec.id, fault_type=t, value=val, metadata=meta)

        raise ValueError(f"Fault '{spec.id}': unknown action.type '{t}'")