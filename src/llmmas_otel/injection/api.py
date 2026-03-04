from __future__ import annotations

from .config import enable_fault_injection_from_file
from .engine import disable_fault_injection as _disable
from .engine import is_enabled as _is_enabled


def enable(path: str, *, seed: str = "0") -> None:
    """
    Enable config-driven fault injection from a YAML/JSON file.

    Example:
      from llmmas_otel.injection import enable
      enable("faults.yaml", seed="demo")
    """
    enable_fault_injection_from_file(path, seed=seed)


def disable() -> None:
    """
    Disable fault injection globally.
    """
    _disable()


def enabled() -> bool:
    """
    Return True if fault injection is enabled.
    """
    return _is_enabled()