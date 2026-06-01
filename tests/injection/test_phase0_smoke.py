from llmmas_otel.injection import (
    DecisionKind,
    FaultSpec,
    HookContext,
    HookType,
    SpecFaultEngine,
)


def test_a2a_truncate_smoke():
    spec = FaultSpec.from_dict({
        "id": "smoke-truncate",
        "hook": "a2a_send",
        "selector": {},
        "action": {"type": "a2a.truncate", "params": {"max_chars": 50}},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(hook_type=HookType.A2A_SEND)
    decision = engine.decide(ctx, payload="x" * 200)

    assert decision.kind == DecisionKind.MUTATE
    assert len(decision.mutated_payload) <= 50
