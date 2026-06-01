# llmmas-otel — Claude Code project memory

Telemetry and fault-injection layer for LLM multi-agent systems. Used in an
ICSE-track study on trace-based anomaly detection in MAS. Faults are injected
into a HyperAgent + SWE-bench Verified runner; resulting OpenTelemetry traces
are consumed by a downstream detector.

## Invariants (never violate)

1. **Feature-leakage rule.** Any span attribute prefixed `llmmas.fault.*` is
   ground-truth label data for the downstream study. Code in this repo MAY
   write those attributes (the injection layer is supposed to). Code outside
   this repo (the detector / feature extraction) MUST NEVER read them. If a
   task description ever asks you to read `llmmas.fault.*` from feature
   extraction code, stop and flag it.

2. **Do not modify shipped action handlers.** These are in production use:
   `a2a.drop`, `a2a.delay`, `a2a.truncate`, `tool.not_installed`,
   `tool.timeout`, `tool.delay`, `tool.malformed_response`, `llm.delay`,
   `llm.rate_limit`, `llm.timeout`, `llm.network_error`,
   `llm.malformed_response`. You may *add* new branches in the same dispatch
   chain; you may not edit the existing ones.

3. **Action-type naming.** `<boundary>.<verb>`, lowercase, dot-separated.
   Examples: `a2a.instruction_loss`, `llm.prompt_inject`, `tool.swap_result`.

4. **Every decision is fully attributed.** Any `InjectionDecision` returned by
   `spec_engine.py` MUST set `fault_id` and `fault_type`. It MUST include a
   `metadata` dict carrying at minimum the original payload hash
   (`original_sha`) or original length (`original_chars`).

5. **No silent defaults.** If `action.params` is missing required input,
   return `InjectionDecision.pass_through()`. Do not invent default payloads.

## Codebase map

- `src/llmmas_otel/injection/spec_engine.py` — action-type dispatch chain;
  primary file for adding new faults.
- `src/llmmas_otel/injection/types.py` — `HookType`, `DecisionKind`,
  `HookContext`, `InjectionDecision`.
- `src/llmmas_otel/injection/spec.py` — YAML/JSON spec loader. Do not modify.
- `src/llmmas_otel/injection/matcher.py` — selector matching. Do not modify.
- `src/llmmas_otel/span_factory.py` — span emission. A2A `MUTATE` already
  wired around line 360 (uses `effective_body` and `apply_mutation`). Do not
  modify unless `FAULT_SPEC.md` explicitly says so.
- `src/llmmas_otel/decorators.py` — LLM and tool call decision consumption.
  Today only consumes `RAISE` and `RETURN`. Phase 3 of `FAULT_SPEC.md`
  extends this; do not touch it before then.
- `src/llmmas_otel/integrations/hyperagent.py` — HyperAgent-specific
  patches. Phase 3 also extends this.
- `tests/injection/` — unit tests for fault handlers. New faults add tests here.

## Commands

- Run all tests:   `pytest -xvs tests/`
- Run one test:    `pytest -xvs tests/injection/test_<name>.py::<test_fn>`
- Install (editable): `pip install -e .`
- Check linters configured in `pyproject.toml` before running any formatter.
  If unsure, ask before running.

## Working agreement with the human

- Implement **exactly one phase per session** (one fault, or one
  infrastructure step). Never batch multiple phases.
- Always show diffs before applying changes; wait for explicit approval.
- After each fault is implemented, run its unit test and report the result
  in chat. Do not commit; the human commits after review.
- If anything in `FAULT_SPEC.md` is ambiguous or contradicts what you see in
  the code, **ask** before guessing. Better to pause than to write 200 lines
  in the wrong direction.
- Do not touch files outside the scope listed in the current task. If a
  refactor seems necessary, propose it in chat; do not do it unilaterally.
- Do not run `git commit`, `git push`, or any destructive command without
  being asked.
