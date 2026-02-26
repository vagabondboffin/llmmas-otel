from __future__ import annotations

from dataclasses import dataclass
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


# Global active engine (set by enable_fault_injection)
_ACTIVE_ENGINE: FaultEngine = NoOpFaultEngine()
_ENABLED: bool = False


def enable_fault_injection(engine: FaultEngine) -> None:
    """
    Enable fault injection using a provided engine.
    (Config-driven enablement comes in M2.1 where we load specs from YAML/JSON.)
    """
    global _ACTIVE_ENGINE, _ENABLED
    _ACTIVE_ENGINE = engine
    _ENABLED = True


def disable_fault_injection() -> None:
    global _ACTIVE_ENGINE, _ENABLED
    _ACTIVE_ENGINE = NoOpFaultEngine()
    _ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


def get_engine() -> FaultEngine:
    return _ACTIVE_ENGINE