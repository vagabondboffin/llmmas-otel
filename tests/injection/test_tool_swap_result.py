from llmmas_otel.injection import (
    DecisionKind,
    FaultSpec,
    HookContext,
    HookType,
    SpecFaultEngine,
)


def test_tool_swap_result_returns_mimicked_tool_output():
    spec = FaultSpec.from_dict({
        "id": "tool-swap-test",
        "hook": "tool_call",
        "selector": {"tool_name": "code_search"},
        "action": {
            "type": "tool.swap_result",
            "params": {
                "mimic": "open_file",
                "value": "contents from the wrong file",
            },
        },
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.TOOL_CALL,
        tool_name="code_search",
    )

    decision = engine.decide(ctx)

    assert decision.kind == DecisionKind.RETURN
    assert decision.return_value == "contents from the wrong file"
    assert decision.fault_id == "tool-swap-test"
    assert decision.fault_type == "tool.swap_result"
    assert decision.metadata["mimicked_tool"] == "open_file"
    assert decision.metadata["actual_tool"] == "code_search"


def test_tool_swap_result_passes_through_without_mimic():
    spec = FaultSpec.from_dict({
        "id": "tool-swap-no-mimic",
        "hook": "tool_call",
        "selector": {"tool_name": "code_search"},
        "action": {
            "type": "tool.swap_result",
            "params": {"value": "fake output"},
        },
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.TOOL_CALL,
        tool_name="code_search",
    )

    decision = engine.decide(ctx)

    assert decision.kind == DecisionKind.PASS


def test_tool_swap_result_passes_through_without_value():
    spec = FaultSpec.from_dict({
        "id": "tool-swap-no-value",
        "hook": "tool_call",
        "selector": {"tool_name": "code_search"},
        "action": {
            "type": "tool.swap_result",
            "params": {"mimic": "open_file"},
        },
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.TOOL_CALL,
        tool_name="code_search",
    )

    decision = engine.decide(ctx)

    assert decision.kind == DecisionKind.PASS
