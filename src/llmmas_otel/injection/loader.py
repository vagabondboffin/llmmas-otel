from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .spec import FaultSpec


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_fault_specs(path: str) -> list[FaultSpec]:
    """
    Load fault specs from YAML/JSON file.

    Supported shapes:
      1) dict with key 'faults': { faults: [ ... ] }
      2) list root: [ ... ]
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fault spec file not found: {path}")

    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        raw = _load_yaml(p)
    elif suffix == ".json":
        raw = _load_json(p)
    else:
        raise ValueError("Fault spec file must end with .yaml/.yml or .json")

    if raw is None:
        return []

    if isinstance(raw, dict) and "faults" in raw:
        faults_raw = raw["faults"]
    else:
        faults_raw = raw

    if not isinstance(faults_raw, list):
        raise ValueError("Fault spec must be a list, or a dict containing key 'faults' as a list")

    specs: list[FaultSpec] = []
    for i, entry in enumerate(faults_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Fault entry at index {i} must be an object/dict")
        specs.append(FaultSpec.from_dict(entry))

    # Ensure IDs unique
    ids = [s.id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Fault IDs must be unique")

    return specs