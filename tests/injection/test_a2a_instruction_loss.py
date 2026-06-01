from llmmas_otel.injection import (
    DecisionKind,
    FaultSpec,
    HookContext,
    HookType,
    SpecFaultEngine,
)


def test_instruction_loss_strips_request_block():
    spec = FaultSpec.from_dict({
        "id": "instruction-loss-test",
        "hook": "a2a_send",
        "selector": {"source_agent_id": "Planner"},
        "action": {"type": "a2a.instruction_loss"},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.A2A_SEND,
        source_agent_id="Planner",
        target_agent_id="Navigator",
    )
    payload = "Context: keep this context\n\nRequest: please find foo() in the repo\n\nEnd."
    decision = engine.decide(ctx, payload=payload)

    assert decision.kind == DecisionKind.MUTATE
    assert "Context: keep this context" in decision.mutated_payload
    assert "[REDACTED]" in decision.mutated_payload
    assert "please find foo()" not in decision.mutated_payload
    assert decision.metadata["original_chars"] == len(payload)
    assert decision.metadata["mutated_chars"] == len(decision.mutated_payload)
    assert decision.metadata["strategy"] == "regex_strip_request_block"
    assert decision.fault_id == "instruction-loss-test"
    assert decision.fault_type == "a2a.instruction_loss"
