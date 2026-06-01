# FAULT_SPEC.md — Fault Injection Implementation Spec

This file is a strict specification. Each fault entry below has a fixed
shape: identifier, hook, selector contract, params schema, required
metadata, handler skeleton, and an acceptance test. Implement against the
spec; do not improvise outside it.

## How to use this file

- Phases run in order. Do not start Phase N+1 until the human has reviewed
  and accepted Phase N.
- Each phase has an **acceptance gate**: a specific pytest command that
  must pass before the phase is considered done.
- Faults marked `STATUS: shipped` already exist in `spec_engine.py`. Phase
  goals for those are *verification only* — no new handlers.
- Faults marked `STATUS: to_implement` require a new branch in the
  dispatch chain of `spec_engine.py`. Use the exact handler skeleton given
  unless it conflicts with code you see in the file (in which case, ask).

## Status table

| #  | Action type              | Boundary | Status        | Phase |
|----|--------------------------|----------|---------------|-------|
| 1  | a2a.instruction_loss     | A2A      | to_implement  | 1     |
| 2  | a2a.hallucinate          | A2A      | to_implement  | 1     |
| 3  | a2a.truncate             | A2A      | shipped       | 2     |
| 4  | a2a.confident_wrong      | A2A      | to_implement  | 1     |
| 5  | llm.malformed_response   | LLM      | shipped       | 5     |
| 6  | llm.prompt_inject        | LLM      | to_implement  | 4     |
| 7  | llm.problem_reframe      | LLM      | to_implement  | 4     |
| 8  | tool.malformed_response  | Tool     | shipped       | 5     |
| 9  | tool.swap_result         | Tool     | to_implement  | 6     |

## MAST mapping (for paper authorship; not needed for implementation)

Reference: Cemri et al. 2025, *Why Do Multi-Agent LLM Systems Fail?*
(arXiv:2503.13657). FM-x.y identifiers refer to MAST failure modes.

- a2a.instruction_loss → FM-2.4 (Information withholding) + FM-1.4 (History loss)
- a2a.hallucinate → induces FM-3.3 (Incorrect verification), FM-1.1 (Disobey task)
- a2a.truncate → FM-1.4 (Loss of conversation history)
- a2a.confident_wrong → FM-3.1 (Premature termination) + FM-3.2 (No verification)
- llm.malformed_response → operational; induces FM-1.2 (Disobey role spec)
- llm.prompt_inject → FM-2.3 (Task derailment)
- llm.problem_reframe → FM-1.1 (Disobey task specification)
- tool.malformed_response → operational; induces FM-1.3 (Step repetition)
- tool.swap_result → FM-2.6 (Reasoning-action mismatch)

---

# Phase 0 — Environment smoke

**Goal.** Verify the fault-injection pipeline produces visible span
attributes end-to-end before any new code is written.

**Tasks.**
1. Create `tests/injection/test_phase0_smoke.py` with a single test that:
   - Constructs a `FaultSpec` with `action.type = "a2a.truncate"` and
     `params: {max_chars: 50}`.
   - Constructs a minimal `HookContext` of `HookType.A2A_SEND`.
   - Calls `SpecFaultEngine(specs=[spec]).decide(ctx, payload="x"*200)`.
   - Asserts the returned `InjectionDecision.kind == DecisionKind.MUTATE`
     and `len(decision.mutated_payload) <= 50`.

**Acceptance gate.**
```
pytest -xvs tests/injection/test_phase0_smoke.py
```

If this fails, the rest of the spec cannot proceed. Report the failure
and stop.

---

# Phase 1 — A2A payload-mutation faults (Faults 1, 2, 4)

**Goal.** Add three new action types that mutate A2A message payloads via
the existing `MUTATE` decision kind. No infrastructure changes. The
`MUTATE` path is already wired in `span_factory.a2a_send` around line 360
(uses `effective_body` and `apply_mutation`).

## Phase 1, Step 1 — shared scaffolding

Create `src/llmmas_otel/injection/fixtures/__init__.py` (empty) and
`src/llmmas_otel/injection/fixtures/a2a_templates.py` with the following
content:

```python
"""Templated A2A payloads for mutation faults, keyed by sender agent role."""

# Plausible-but-false structured claims that name a specific function /
# file / test count. Use for `a2a.hallucinate`.
HALLUCINATION_TEMPLATES = {
    "navigator": (
        "Final answer: Found function `process_request` at "
        "src/core/handler.py:142. It calls `validate_input` on line 156."
    ),
    "editor": (
        "Patch applied successfully to src/core/handler.py. "
        "The flag `strict_mode` was changed from False to True."
    ),
    "executor": (
        "All tests passed. test_handler.py: 14 passed, 0 failed in 2.3s."
    ),
}

# Confident task-complete claims without specifics. Use for
# `a2a.confident_wrong`.
CONFIDENT_SUCCESS_TEMPLATES = {
    "navigator": "Final answer: All required context located. Task complete.",
    "editor":    "All requested edits applied. Patch validated. Task complete.",
    "executor":  "Tests executed successfully. All assertions passed. Task complete.",
}
```

**Note on role keys.** Before considering Phase 1 complete, grep the real
HyperAgent integration for the actual `source_agent_id` strings used at
A2A boundaries (likely in `integrations/hyperagent.py` near
`default_span_factory.a2a_send`). If the real strings differ from
`navigator` / `editor` / `executor`, update the template keys to match
exactly (case-sensitive). Report the keys you found.

## Phase 1, Step 2 — Fault 1: a2a.instruction_loss

```yaml
action_type: a2a.instruction_loss
hook: a2a_send
selector_supported: [source_agent_id, target_agent_id, phase_order, step_index]
params: {}   # no params required
required_metadata: [original_chars, mutated_chars, strategy]
```

**Handler — add as new branch in `spec_engine.py` dispatch chain:**

```python
if t == "a2a.instruction_loss":
    if payload is None:
        return InjectionDecision.pass_through()
    # HyperAgent A2A messages carry an instruction block labelled
    # "Request:" — strip it, keep the surrounding Context block.
    import re
    mutated = re.sub(
        r"Request\s*:\s*.*?(?=(\n\s*\n|\Z))",
        "Request: [REDACTED]",
        payload,
        flags=re.DOTALL | re.IGNORECASE,
    )
    meta = {
        "original_chars": len(payload),
        "mutated_chars": len(mutated),
        "strategy": "regex_strip_request_block",
    }
    return InjectionDecision.mutate(
        fault_id=spec.id, fault_type=t,
        mutated_payload=mutated, metadata=meta,
    )
```

**Unit test — `tests/injection/test_a2a_instruction_loss.py`:**

```python
def test_instruction_loss_strips_request_block():
    spec = FaultSpec.from_dict({
        "id": "t",
        "hook": "a2a_send",
        "selector": {"source_agent_id": "planner"},
        "action": {"type": "a2a.instruction_loss"},
    })
    engine = SpecFaultEngine(specs=[spec])
    ctx = HookContext(
        hook_type=HookType.A2A_SEND,
        source_agent_id="planner",
        target_agent_id="navigator",
    )
    payload = "Context: blah blah\n\nRequest: please find foo() in the repo\n\nEnd."
    decision = engine.decide(ctx, payload=payload)
    assert decision.kind == DecisionKind.MUTATE
    assert "[REDACTED]" in decision.mutated_payload
    assert "please find foo()" not in decision.mutated_payload
    assert decision.metadata["original_chars"] == len(payload)
    assert decision.fault_type == "a2a.instruction_loss"
```

## Phase 1, Step 3 — Fault 2: a2a.hallucinate

```yaml
action_type: a2a.hallucinate
hook: a2a_send
selector_supported: [source_agent_id, target_agent_id, phase_order, step_index]
params: {}
required_metadata: [template_role, original_sha]
```

**Handler:** import templates at top of `spec_engine.py`:
```python
from .fixtures.a2a_templates import HALLUCINATION_TEMPLATES, CONFIDENT_SUCCESS_TEMPLATES
import hashlib
def _sha(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()
```

Add dispatch branch:
```python
if t == "a2a.hallucinate":
    if payload is None:
        return InjectionDecision.pass_through()
    sender = ctx.source_agent_id or ""
    template = HALLUCINATION_TEMPLATES.get(sender)
    if template is None:
        return InjectionDecision.pass_through()
    meta = {"template_role": sender, "original_sha": _sha(payload)}
    return InjectionDecision.mutate(
        fault_id=spec.id, fault_type=t,
        mutated_payload=template, metadata=meta,
    )
```

**Unit test — `tests/injection/test_a2a_hallucinate.py`:** assert decision
is MUTATE, `mutated_payload == HALLUCINATION_TEMPLATES["navigator"]`,
`metadata["template_role"] == "navigator"`, and `original_sha` is a
64-char hex string.

## Phase 1, Step 4 — Fault 4: a2a.confident_wrong

Identical handler shape to Fault 2 but uses
`CONFIDENT_SUCCESS_TEMPLATES`. Either duplicate or factor a helper
`_a2a_template_mutate(spec, ctx, payload, templates, action_type)` and
call it from both branches. Unit test mirrors Fault 2.

## Phase 1 acceptance gate

```
pytest -xvs tests/injection/test_a2a_instruction_loss.py \
            tests/injection/test_a2a_hallucinate.py \
            tests/injection/test_a2a_confident_wrong.py
```

All three tests pass. The dispatch chain in `spec_engine.py` contains
exactly three new branches (plus the import line). No other file changed.

---

# Phase 2 — Verify Fault 3 (a2a.truncate, shipped)

**Goal.** Confirm the shipped `a2a.truncate` action behaves as documented
under our test harness. No new handler.

**Task.** Add `tests/injection/test_a2a_truncate.py` exercising the
existing branch with payload longer than `max_chars`; assert decision is
MUTATE and `len(mutated_payload) <= max_chars`. Read the existing
`a2a.truncate` branch in `spec_engine.py` first to confirm parameter name.

**Acceptance gate.**
```
pytest -xvs tests/injection/test_a2a_truncate.py
```

---

# Phase 3 — Input-mutation infrastructure (one-time)

**Goal.** Extend the LLM-call and tool-call decision paths to support
mutation of *inputs* (messages / args) before the wrapped function runs.
Currently they consume only `RAISE` and `RETURN`. This phase adds zero
new faults; it unlocks Phase 4 and the optional variant of Phase 6.

## Phase 3, Step 1 — extend `DecisionKind`

In `src/llmmas_otel/injection/types.py`:

- Add `MUTATE_INPUT = "mutate_input"` to the `DecisionKind` enum.
- Add a factory:
  ```python
  @staticmethod
  def mutate_input(*, fault_id, fault_type, mutator, metadata=None):
      """`mutator`: callable taking (args, kwargs) and returning the same shape."""
      return InjectionDecision(
          kind=DecisionKind.MUTATE_INPUT,
          fault_id=fault_id, fault_type=fault_type,
          return_value=mutator,        # reused field
          metadata=metadata or {},
      )
  ```

## Phase 3, Step 2 — extend LLM-call decorator

In `src/llmmas_otel/decorators.py`, inside the LLM-call wrapper (find the
existing `if dec.kind == DecisionKind.RAISE:` block; it is preceded by
`dec = default_span_factory.current_llm_call_decision()`):

Insert *before* the existing RAISE/RETURN handling:
```python
if dec is not None and dec.kind == DecisionKind.MUTATE_INPUT:
    mutator = dec.return_value
    try:
        args, kwargs = mutator(args, kwargs)
    except Exception as e:
        ctx.span.set_attribute("llmmas.fault.error", f"mutator_failed: {e}")
```

After this branch the normal `fn(*args, **kwargs)` call proceeds.

## Phase 3, Step 3 — extend tool-call decision

In `src/llmmas_otel/decorators.py::_run_with_tool_fault_decision`,
identical pattern: before existing RAISE/RETURN, insert:
```python
if dec is not None and dec.kind == DecisionKind.MUTATE_INPUT:
    args, kwargs = dec.return_value(args, kwargs)
```

## Phase 3, Step 4 — mirror in HyperAgent integration

In `src/llmmas_otel/integrations/hyperagent.py`, locate
`_patch_autogen_llm_calls`. The wrapped call (`OpenAIClient.create` or
similar) needs the same `MUTATE_INPUT` branch *before* the call so the
mutator runs on the `messages` list that's actually sent to the provider.
Apply the same pattern as Phase 3 Step 2.

## Phase 3 acceptance gate

Add `tests/injection/test_mutate_input.py`:
- Build a `FaultSpec` whose handler returns `mutate_input(...)` with a
  trivial mutator that uppercases the first user message.
- Wrap a fake `fn(messages)` with the LLM decorator.
- Invoke the wrapper with a `messages` list containing one user turn.
- Assert the wrapped fn received the uppercased message.

```
pytest -xvs tests/injection/test_mutate_input.py
```

---

# Phase 4 — LLM input mutation faults (Faults 6, 7)

Both use the Phase 3 machinery.

## Phase 4, Step 1 — Fault 6: llm.prompt_inject

```yaml
action_type: llm.prompt_inject
hook: llm_call
selector_supported: [agent_id, phase_order, step_index]
params:
  note: str   # the system note to append; required
required_metadata: [note_chars]
```

**Handler in `spec_engine.py`:**
```python
if t == "llm.prompt_inject":
    note = spec.action.params.get("note")
    if not note:
        return InjectionDecision.pass_through()
    def mutator(args, kwargs):
        msgs = kwargs.get("messages") or (args[0] if args else None)
        if not isinstance(msgs, list):
            return args, kwargs
        msgs = list(msgs) + [{"role": "system", "content": note}]
        if "messages" in kwargs:
            kwargs["messages"] = msgs
        else:
            args = (msgs,) + args[1:]
        return args, kwargs
    return InjectionDecision.mutate_input(
        fault_id=spec.id, fault_type=t, mutator=mutator,
        metadata={"note_chars": len(note)},
    )
```

**Unit test:** build a fake `messages` list with one user turn, call the
decision's mutator on `(args, kwargs)`, assert the result's `messages`
ends with a `{"role": "system", "content": <note>}` entry.

## Phase 4, Step 2 — Fault 7: llm.problem_reframe

```yaml
action_type: llm.problem_reframe
hook: llm_call
selector_supported: [agent_id, phase_order, step_index]
params:
  alt_problem: str   # the reframed task; required
required_metadata: [alt_chars]
```

**Handler:** same shape as Fault 6 but mutates the first `user`-role
message by prepending `"Reinterpret the issue as: {alt_problem}\n\n"`.
Returns `mutate_input(...)` with metadata `{"alt_chars": len(alt_problem)}`.

**Unit test:** assert the first user message after mutation begins with
`"Reinterpret the issue as: "` and the original content is preserved
after the prefix.

## Phase 4 acceptance gate

```
pytest -xvs tests/injection/test_llm_prompt_inject.py \
            tests/injection/test_llm_problem_reframe.py
```

---

# Phase 5 — Verify Faults 5, 8 (shipped)

**Goal.** Confirm `llm.malformed_response` and `tool.malformed_response`
behave as documented.

**Task.** Two new tests:
- `tests/injection/test_llm_malformed_shipped.py` — build spec with
  `action: {type: llm.malformed_response, params: {value: "{invalid"}}`,
  decide on a fake LLM_CALL context, assert decision is RETURN with the
  configured value.
- `tests/injection/test_tool_malformed_shipped.py` — same shape on
  TOOL_CALL.

**Acceptance gate.**
```
pytest -xvs tests/injection/test_llm_malformed_shipped.py \
            tests/injection/test_tool_malformed_shipped.py
```

---

# Phase 6 — Fault 9: tool.swap_result

```yaml
action_type: tool.swap_result
hook: tool_call
selector_supported: [tool_name, agent_id, step_index]
params:
  mimic: str    # name of tool whose output shape is being mimicked; required
  value: str    # the output string to return; required
required_metadata: [mimicked_tool, actual_tool]
```

**Rationale.** The agent calls tool A; the system returns content shaped
like tool B's output. The trace shows the call site for A but the
content for B. This induces MAST FM-2.6 (reasoning-action mismatch).

**Handler in `spec_engine.py`:**
```python
if t == "tool.swap_result":
    mimic = spec.action.params.get("mimic")
    value = spec.action.params.get("value")
    if not mimic or value is None:
        return InjectionDecision.pass_through()
    return InjectionDecision.return_(
        fault_id=spec.id, fault_type=t, value=value,
        metadata={"mimicked_tool": mimic, "actual_tool": ctx.tool_name},
    )
```

**Unit test:** build a HookContext with `tool_name="code_search"`,
assert decision is RETURN with `metadata["mimicked_tool"] == "open_file"`.

**Acceptance gate.**
```
pytest -xvs tests/injection/test_tool_swap_result.py
```

---

# Final acceptance — all phases

```
pytest -xvs tests/injection/
```

All tests pass. No `llmmas.fault.*` attribute reads exist anywhere outside
`src/llmmas_otel/`. Every new branch in `spec_engine.py` returns a
decision with both `fault_id` and `fault_type` populated and a non-empty
`metadata` dict.
