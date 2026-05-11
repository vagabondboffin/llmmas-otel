from __future__ import annotations

from .config import enable_fault_injection_from_file
from .engine import (
    disable_fault_injection as _disable,
    is_enabled as _is_enabled,
    is_fault_trace_visible as _is_fault_trace_visible,
    set_fault_trace_visibility as _set_fault_trace_visibility,
)


def enable(path: str, *, seed: str = "0", trace_visible: bool = True) -> None:
    """
    Enable config-driven fault injection from a YAML/JSON file.
    """
    enable_fault_injection_from_file(path, seed=seed, trace_visible=trace_visible)


def disable() -> None:
    _disable()


def enabled() -> bool:
    return _is_enabled()


def set_trace_visibility(visible: bool) -> None:
    _set_fault_trace_visibility(visible)


def trace_visible() -> bool:
    return _is_fault_trace_visible()