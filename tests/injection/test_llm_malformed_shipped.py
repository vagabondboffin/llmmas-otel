from llmmas_otel.injection import (
    DecisionKind,
    FaultSpec,
    HookContext,
    HookType,
    SpecFaultEngine,
)


def test_llm_malformed_response_returns_supplied_value():
    spec = FaultSpec.from_dict({
        "id": "llm-malformed-test",
        "hook": "llm_call",
        "selector": {"agent_id": "Planner"},
        "action": {"type": "llm.malformed_response", "params": {"value": "{invalid"}},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.LLM_CALL,
        agent_id="Planner",
    )

    decision = engine.decide(ctx)

    assert decision.kind == DecisionKind.RETURN
    assert decision.return_value == "{invalid"
    assert decision.fault_id == "llm-malformed-test"
    assert decision.fault_type == "llm.malformed_response"
    assert decision.metadata == {"returned_type": "str"}
