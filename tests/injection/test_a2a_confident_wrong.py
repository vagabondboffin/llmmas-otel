from llmmas_otel.injection import (
    DecisionKind,
    FaultSpec,
    HookContext,
    HookType,
    SpecFaultEngine,
)
from llmmas_otel.injection.fixtures.a2a_templates import CONFIDENT_SUCCESS_TEMPLATES


def test_a2a_confident_wrong_replaces_payload_with_success_template():
    spec = FaultSpec.from_dict({
        "id": "confident-wrong-test",
        "hook": "a2a_send",
        "selector": {"source_agent_id": "Executor"},
        "action": {"type": "a2a.confident_wrong"},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.A2A_SEND,
        source_agent_id="Executor",
        target_agent_id="Planner",
    )
    payload = "Real Executor result: tests failed with assertion error."
    decision = engine.decide(ctx, payload=payload)

    assert decision.kind == DecisionKind.MUTATE
    assert decision.mutated_payload == CONFIDENT_SUCCESS_TEMPLATES["Executor"]
    assert payload not in decision.mutated_payload
    assert "Task complete" in decision.mutated_payload
    assert decision.metadata["template_role"] == "Executor"
    assert len(decision.metadata["original_sha"]) == 64
    assert all(c in "0123456789abcdef" for c in decision.metadata["original_sha"])
    assert decision.fault_id == "confident-wrong-test"
    assert decision.fault_type == "a2a.confident_wrong"


def test_a2a_confident_wrong_passes_through_for_unknown_sender():
    spec = FaultSpec.from_dict({
        "id": "confident-wrong-unknown-sender",
        "hook": "a2a_send",
        "selector": {},
        "action": {"type": "a2a.confident_wrong"},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.A2A_SEND,
        source_agent_id="Planner",
        target_agent_id="Navigator",
    )
    decision = engine.decide(ctx, payload="Planner message")

    assert decision.kind == DecisionKind.PASS
