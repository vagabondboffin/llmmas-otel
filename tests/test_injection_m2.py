from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from llmmas_otel.injection import (
    HookContext,
    HookType,
    DecisionKind,
    selector_matches,
    FaultSelector,
    load_fault_specs,
    SpecFaultEngine,
    enable_fault_injection_from_file,
    disable_fault_injection,
    get_engine,
)


class TestInjectionM2(unittest.TestCase):
    def tearDown(self) -> None:
        # Ensure global injection state does not leak across tests
        disable_fault_injection()

    def test_selector_matches_basic(self) -> None:
        sel = FaultSelector.from_dict(
            {
                "phase_name": "Coding",
                "source_agent_id": "Planner",
                "target_agent_id": "Coder",
            }
        )
        ctx_ok = HookContext(
            hook_type=HookType.A2A_SEND,
            session_id="S1",
            phase_name="Coding",
            source_agent_id="Planner",
            target_agent_id="Coder",
        )
        ctx_bad = HookContext(
            hook_type=HookType.A2A_SEND,
            session_id="S1",
            phase_name="Planning",
            source_agent_id="Planner",
            target_agent_id="Coder",
        )
        self.assertTrue(selector_matches(sel, ctx_ok))
        self.assertFalse(selector_matches(sel, ctx_bad))

    def test_max_times_enforced(self) -> None:
        # max_times = 1 => applies only once per session
        yaml_text = """
faults:
  - id: FMAX
    hook: a2a_send
    selector:
      phase_name: Coding
    action:
      type: a2a.drop
    limits:
      probability: 1.0
      max_times: 1
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "faults.yaml"
            p.write_text(yaml_text, encoding="utf-8")
            specs = load_fault_specs(str(p))
            engine = SpecFaultEngine(specs=specs, seed="0")

            ctx = HookContext(hook_type=HookType.A2A_SEND, session_id="S1", phase_name="Coding")
            d1 = engine.decide(ctx, payload="hello")
            d2 = engine.decide(ctx, payload="hello")

            self.assertEqual(d1.kind, DecisionKind.DROP)
            self.assertEqual(d2.kind, DecisionKind.PASS)

    def test_probability_zero_never_applies(self) -> None:
        yaml_text = """
faults:
  - id: FPROB0
    hook: a2a_send
    selector:
      phase_name: Coding
    action:
      type: a2a.drop
    limits:
      probability: 0.0
      max_times: 10
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "faults.yaml"
            p.write_text(yaml_text, encoding="utf-8")
            specs = load_fault_specs(str(p))
            engine = SpecFaultEngine(specs=specs, seed="0")

            ctx = HookContext(hook_type=HookType.A2A_SEND, session_id="S1", phase_name="Coding")
            d = engine.decide(ctx, payload="hello")
            self.assertEqual(d.kind, DecisionKind.PASS)

    def test_a2a_truncate_mutate(self) -> None:
        yaml_text = """
faults:
  - id: FTRUNC
    hook: a2a_send
    selector:
      phase_name: Coding
    action:
      type: a2a.truncate
      params:
        max_chars: 5
    limits:
      probability: 1.0
      max_times: 10
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "faults.yaml"
            p.write_text(yaml_text, encoding="utf-8")
            specs = load_fault_specs(str(p))
            engine = SpecFaultEngine(specs=specs, seed="0")

            payload = "abcdefghijklmnopqrstuvwxyz"
            ctx = HookContext(hook_type=HookType.A2A_SEND, session_id="S1", phase_name="Coding")
            d = engine.decide(ctx, payload=payload)

            self.assertEqual(d.kind, DecisionKind.MUTATE)
            self.assertEqual(d.fault_id, "FTRUNC")
            self.assertEqual(d.fault_type, "a2a.truncate")
            self.assertEqual(d.mutated_payload, payload[:5])
            self.assertEqual(d.metadata.get("original_len"), len(payload))
            self.assertEqual(d.metadata.get("new_len"), 5)

    def test_tool_not_installed_decision(self) -> None:
        yaml_text = """
faults:
  - id: FTOOL
    hook: tool_call
    selector:
      tool_name: static_checker
    action:
      type: tool.not_installed
    limits:
      probability: 1.0
      max_times: 1
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "faults.yaml"
            p.write_text(yaml_text, encoding="utf-8")
            specs = load_fault_specs(str(p))
            engine = SpecFaultEngine(specs=specs, seed="0")

            ctx = HookContext(hook_type=HookType.TOOL_CALL, session_id="S1", tool_name="static_checker")
            d = engine.decide(ctx, payload="--help")

            self.assertEqual(d.kind, DecisionKind.RAISE)
            self.assertEqual(d.fault_id, "FTOOL")
            self.assertEqual(d.fault_type, "tool.not_installed")
            self.assertIsInstance(d.raise_exception, FileNotFoundError)

    def test_enable_from_file_sets_global_engine(self) -> None:
        yaml_text = """
faults:
  - id: FGLOBAL
    hook: a2a_send
    selector:
      phase_name: Coding
    action:
      type: a2a.drop
    limits:
      probability: 1.0
      max_times: 1
"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "faults.yaml"
            p.write_text(yaml_text, encoding="utf-8")

            enable_fault_injection_from_file(str(p), seed="demo")

            engine = get_engine()
            ctx = HookContext(hook_type=HookType.A2A_SEND, session_id="S1", phase_name="Coding")
            d = engine.decide(ctx, payload="hello")

            self.assertEqual(d.kind, DecisionKind.DROP)
            self.assertEqual(d.fault_id, "FGLOBAL")


if __name__ == "__main__":
    unittest.main()