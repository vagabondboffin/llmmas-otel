from llmmas_otel.injection import (
    DecisionKind,
    FaultSpec,
    HookContext,
    HookType,
    SpecFaultEngine,
)


def test_tool_malformed_response_returns_supplied_value():
    spec = FaultSpec.from_dict({
        "id": "tool-malformed-test",
        "hook": "tool_call",
        "selector": {"tool_name": "code_search"},
        "action": {"type": "tool.malformed_response", "params": {"value": "{invalid_tool_output"}},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.TOOL_CALL,
        tool_name="code_search",
    )

    decision = engine.decide(ctx)

    assert decision.kind == DecisionKind.RETURN
    assert decision.return_value == "{invalid_tool_output"
    assert decision.fault_id == "tool-malformed-test"
    assert decision.fault_type == "tool.malformed_response"
    assert decision.metadata == {"returned_type": "str"}
