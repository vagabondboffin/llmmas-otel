from __future__ import annotations

from .engine import enable_fault_injection as _enable_engine
from .loader import load_fault_specs
from .spec_engine import SpecFaultEngine


def enable_fault_injection_from_file(path: str, *, seed: str = "0") -> None:
    """
    Load YAML/JSON fault specs from file and enable injection globally.
    """
    specs = load_fault_specs(path)
    engine = SpecFaultEngine(specs=specs, seed=seed)
    _enable_engine(engine)