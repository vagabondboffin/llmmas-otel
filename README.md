# llmmas-otel

Framework-agnostic **OpenTelemetry (OTel) observability + fault injection hooks** for **LLM-based Multi-Agent Systems (LLM-MAS)**, with an emphasis on **Software Engineering (SE) agentic systems** (e.g., ChatDev, HyperAgent).  
The goal is to provide a lightweight library of **Python decorators/wrappers** that instrument a MAS with a **structured trace hierarchy** (session → segment → agent_step → A2A/tool/LLM) aligned with our observability proposal, and to later support **fault injection** and **trace-based analysis**.

---

## Project phases (milestones) and deliverables

### Milestone 0 — Trace shape MVP (toy demo)
**Goal:** A tiny runnable example that produces the minimal trace hierarchy and core MAS attributes.

**Deliverables:**
- Minimal package skeleton (`src/llmmas_otel/`)
- Console tracing bootstrap (`init_console_tracing()`)
- Decorators:
  - `@observe_session`
  - `@observe_agent_step`
  - `@observe_a2a_send`
- `examples/m0_demo.py` that outputs nested spans:
  - `llmmas.session`
    - `agent_step`
      - `http.client` (A2A send span)
- Core semantic attribute keys (session/agent/step/message/edge identifiers)

---

### Milestone 1 — Proposal-complete observability (framework-agnostic)
**Goal:** Express the full observability structure from the proposal using a small set of wrapper points on any MAS.

**Deliverables:**
- Add `@observe_segment` (phase/segment span)
- Add `@observe_a2a_receive` (receive span)
- Add `@observe_tool_call` (tool span)
- A second runnable example showing:
  - session → segment → agent_step → A2A send/receive → tool
- Minimal “integration recipe” docs: where to place decorators in a generic MAS runner

---

### Milestone 2 — Fault injection overlay (config-driven)
**Goal:** Apply controlled perturbations at boundary points (A2A/tool/LLM wrappers) and record them in traces.

**Deliverables:**
- Fault spec format (YAML/JSON)
- Injection engine (matching rules → apply fault)
- Fault types: (yet to be decided)
- Trace annotations for injected faults (attributes + events, e.g., `fault.applied`)
- Demo: run the same toy MAS with faults on/off and observe trace differences

---

### Milestone 3 — SE LLM-MAS integrations (show portability)
**Goal:** Demonstrate that the library generalizes to real SE agentic systems. (There’s no shame in aiming big :))

**Deliverables:**
- ChatDev integration recipe (minimal patch points + demo run)
- HyperAgent integration recipe (or any alternative SE LLM-MAS)

---

### Milestone 4 — Release (for public use + submission)
**Goal:** Make the tool easy to install, run, and evaluate (paper/demo ready).

**Deliverables:**
- Packaging polish
- Documentation:
  - Quickstart
  - Concepts (span hierarchy + attributes)
  - Integration guides
  - Fault injection guide

---

## Current status

✅ **Milestone 0 completed**  
- Minimal `llmmas-otel` package skeleton created  
- Console exporter demo implemented  
- Session → agent_step → A2A-send trace hierarchy working on a toy example  
- Core semantic attributes defined (session, agent, step, edge, message identifiers)

✅ **Milestone 1 completed**  
- Session → segment/phase → agent_step → send/process → execute_tool
- Instruction for using the observability framework
- new demo 



