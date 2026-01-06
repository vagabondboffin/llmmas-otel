# llmmas-otel integration recipe (framework-agnostic)

This document explains **where to place llmmas-otel wrappers** in any LLM-MAS runtime so you get the standard trace structure:

`llmmas.session → llmmas.workflow.segment → agent_step → (send/process) → execute_tool`

The key idea is to instrument only **stable chokepoints** (runner, agent step, message send/receive, tool executor) rather than sprinkling spans everywhere.

---

## 0) Prerequisites (OTel configuration)

`llmmas-otel` emits spans using the OpenTelemetry API. Your application must configure a tracer provider/exporter (e.g., OTLP → Jaeger/Collector) **or** use the demo bootstrap in `llmmas_otel.bootstrap`.

- Production / existing OTel setup: configure OTel in your app (recommended).
- Demo: use `llmmas_otel.bootstrap.init_otlp_tracing(...)`.

---

## 1) Identify the 6 chokepoints in your MAS

### 1. Runner / top-level task (Session)
**Goal:** exactly one root span per task/run.

Wrap the top-level function that runs a single task:

- `run_task(task)` / `app.invoke(input)` / `runner.run(...)`

Use:
- `@observe_session(session_id=...)`

This produces the root span:
- `llmmas.session`

---

### 2. Phase/segment boundary (Segment)
**Goal:** represent phases like planning / implementation / testing.

Wrap:
- the function that runs a phase, OR
- the phase loop body.

Use:
- `@observe_segment(name="planning", order=0)` or `with segment(name="planning", order=0): ...`

This produces:
- `llmmas.workflow.segment`

---

### 3. Agent step boundary (Agent Step)
**Goal:** one span per agent turn/step.

Wrap the smallest function that represents one agent step:
- `agent.step(...)`
- `agent.act(...)`
- `on_turn(...)`

Use:
- `@observe_agent_step(agent_id="Planner", step_index=0)`

This produces:
- `agent_step`

---

### 4. Message dispatch boundary (A2A Send)
**Goal:** span for outgoing message + inject context into message headers/metadata.

Wrap the function that **enqueues/sends** the message:
- `bus.send(envelope)`
- `mailbox.put(msg)`
- `router.dispatch(msg)`

Use:
- `@observe_a2a_send(...)`

Provide:
- `carrier_fn`: returns a mutable dict on the message/envelope where trace context can be injected (e.g., `envelope.headers`).
- `message_body_fn`: returns the text body (optional; used for preview/hash + message store).

This produces:
- `send {edge_id}`

---

### 5. Message handler entry (A2A Receive / Process)
**Goal:** span for processing an incoming message + link back to the send span.

Wrap the function where the message **enters** the target agent:
- mailbox dequeue handler
- router dispatch to agent handler
- `agent.on_message(msg)`

Use:
- `@observe_a2a_receive(...)`

Provide:
- `carrier_fn`: returns the dict that contains injected context (same field used on send).

This produces:
- `process {edge_id}`

In the trace, it appears as:
- a child of the receiver's `agent_step`
- with a reference/link to the corresponding `send` span

---

### 6. Tool executor chokepoint (Tool Call)
**Goal:** span for tool execution (execute_tool).

Wrap the single function that all tools go through:
- `tool_executor.execute(tool_name, args)`
- `tool.run(...)` if it is the universal entry

Use:
- `@observe_tool_call(tool_name=..., tool_type=..., call_id=...)`

This produces:
- `execute_tool {gen_ai.tool.name}`

By default, keep tool arguments/results out of traces; use preview/hash if needed.

---

## 2) Minimal pseudo-code skeleton

```python
@observe_session(session_id=task.id)
def run_task(task):
    for segment_index, segment_name in enumerate(["planning", "implementation", "testing"]):
        with segment(name=segment_name, order=segment_index):
            while not done:
                agent.step()
```
Message bus boundaries:
```python
@observe_a2a_send(
    source_agent_id="Planner", target_agent_id="Coder", edge_id="Planner->Coder",
    message_id="...", carrier_fn=lambda env: env.headers, message_body_fn=lambda env: env.body
)
def bus_send(env): ...

@observe_a2a_receive(
    source_agent_id="Planner", target_agent_id="Coder", edge_id="Planner->Coder",
    message_id="...", carrier_fn=lambda env: env.headers, message_body_fn=lambda env: env.body
)
def on_message(env): ...
```

Tool boundary:
```python
@observe_tool_call(tool_name="search", tool_type="web")
def execute_tool(args): ...
```