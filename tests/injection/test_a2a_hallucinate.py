from llmmas_otel.injection import (
    DecisionKind,
    FaultSpec,
    HookContext,
    HookType,
    SpecFaultEngine,
)
from llmmas_otel.injection.fixtures.a2a_templates import HALLUCINATION_TEMPLATES


def test_a2a_hallucinate_replaces_payload_with_role_template():
    spec = FaultSpec.from_dict({
        "id": "hallucinate-test",
        "hook": "a2a_send",
        "selector": {"source_agent_id": "Navigator"},
        "action": {"type": "a2a.hallucinate"},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.A2A_SEND,
        source_agent_id="Navigator",
        target_agent_id="Planner",
    )
    payload = "Real Navigator result: I found only partial context."
    decision = engine.decide(ctx, payload=payload)

    assert decision.kind == DecisionKind.MUTATE
    assert decision.mutated_payload == HALLUCINATION_TEMPLATES["Navigator"]
    assert payload not in decision.mutated_payload
    assert decision.metadata["template_role"] == "Navigator"
    assert len(decision.metadata["original_sha"]) == 64
    assert all(c in "0123456789abcdef" for c in decision.metadata["original_sha"])
    assert decision.fault_id == "hallucinate-test"
    assert decision.fault_type == "a2a.hallucinate"


def test_a2a_hallucinate_passes_through_for_unknown_sender():
    spec = FaultSpec.from_dict({
        "id": "hallucinate-unknown-sender",
        "hook": "a2a_send",
        "selector": {},
        "action": {"type": "a2a.hallucinate"},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.A2A_SEND,
        source_agent_id="Planner",
        target_agent_id="Navigator",
    )
    decision = engine.decide(ctx, payload="Planner message")

    assert decision.kind == DecisionKind.PASS
