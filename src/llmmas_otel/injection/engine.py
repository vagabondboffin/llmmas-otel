from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Optional

from .types import HookContext, InjectionDecision


class FaultEngine:
    """
    Interface for fault injection engines.
    """

    def decide(self, ctx: HookContext, payload: Optional[str] = None) -> InjectionDecision:
        raise NotImplementedError


@dataclass
class NoOpFaultEngine(FaultEngine):
    """
    Default engine: never injects faults.
    """

    def decide(self, ctx: HookContext, payload: Optional[str] = None) -> InjectionDecision:
        return InjectionDecision.pass_through()


_ACTIVE_ENGINE: FaultEngine = NoOpFaultEngine()
_ENABLED: bool = False
_FAULT_TRACE_VISIBLE: bool = True


def enable_fault_injection(
    engine_or_path: FaultEngine | str | PathLike[str],
    *,
    seed: str = "0",
    trace_visible: bool = True,
) -> None:
    """
    Enable fault injection globally.

    Accepts either:
      - a FaultEngine instance
      - a YAML/JSON path string, which will be loaded into a SpecFaultEngine

    trace_visible controls whether injected faults are explicitly shown in traces
    via llmmas.fault.* attributes and the fault.applied event.
    """
    global _ACTIVE_ENGINE, _ENABLED, _FAULT_TRACE_VISIBLE

    _FAULT_TRACE_VISIBLE = trace_visible

    if isinstance(engine_or_path, (str, PathLike)):
        from .config import enable_fault_injection_from_file

        enable_fault_injection_from_file(
            str(engine_or_path),
            seed=seed,
            trace_visible=trace_visible,
        )
        return

    if not isinstance(engine_or_path, FaultEngine):
        raise TypeError(
            "enable_fault_injection(...) expects either a FaultEngine instance "
            "or a YAML/JSON config path"
        )

    _ACTIVE_ENGINE = engine_or_path
    _ENABLED = True


def disable_fault_injection() -> None:
    global _ACTIVE_ENGINE, _ENABLED
    _ACTIVE_ENGINE = NoOpFaultEngine()
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def get_engine() -> FaultEngine:
    return _ACTIVE_ENGINE


def set_fault_trace_visibility(visible: bool) -> None:
    global _FAULT_TRACE_VISIBLE
    _FAULT_TRACE_VISIBLE = bool(visible)


def is_fault_trace_visible() -> bool:
    return _FAULT_TRACE_VISIBLE