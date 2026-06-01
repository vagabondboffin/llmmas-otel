from llmmas_otel.injection import (
    DecisionKind,
    FaultSpec,
    HookContext,
    HookType,
    SpecFaultEngine,
)


def test_a2a_truncate_mutates_payload_to_max_chars():
    spec = FaultSpec.from_dict({
        "id": "truncate-test",
        "hook": "a2a_send",
        "selector": {"source_agent_id": "Planner"},
        "action": {"type": "a2a.truncate", "params": {"max_chars": 20}},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.A2A_SEND,
        source_agent_id="Planner",
        target_agent_id="Navigator",
    )
    payload = "Context: this message is intentionally longer than twenty characters."
    decision = engine.decide(ctx, payload=payload)

    assert decision.kind == DecisionKind.MUTATE
    assert decision.mutated_payload == payload[:20]
    assert len(decision.mutated_payload) <= 20
    assert decision.fault_id == "truncate-test"
    assert decision.fault_type == "a2a.truncate"
    assert decision.metadata["original_len"] == len(payload)
    assert decision.metadata["new_len"] == 20
    assert decision.metadata["max_chars"] == 20
