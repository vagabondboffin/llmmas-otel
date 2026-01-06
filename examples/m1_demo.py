from __future__ import annotations

from dataclasses import dataclass, field

from llmmas_otel.bootstrap import init_otlp_tracing
from llmmas_otel import (
    enable_message_store,
    observe_session,
    segment,
    observe_agent_step,
    observe_a2a_send,
    observe_a2a_receive,
    observe_tool_call,
)


enable_message_store("out/messages_m1.jsonl")


@dataclass
class Envelope:
    message_id: str
    body: str
    channel: str = "requirements"
    headers: dict[str, str] = field(default_factory=dict)


# In a real MAS, this would be a queue/mailbox.
INBOX: list[Envelope] = []


@observe_a2a_send(
    source_agent_id="Planner",
    target_agent_id="Coder",
    edge_id="Planner->Coder",
    message_id="msg-0001",
    channel="requirements",
    message_body_fn=lambda env: env.body,
    carrier_fn=lambda env: env.headers,  # <-- inject trace context into headers
)
def send_message(env: Envelope) -> None:
    INBOX.append(env)
    print("SEND:", env.body)


@observe_tool_call(
    tool_name="code_search",
    tool_type="search",
    tool_args_fn=lambda query: query,
    tool_result_fn=lambda result: result,
    record_args=True,
    record_result=True,
)
def tool_code_search(query: str) -> str:
    # fake it till make it
    return f"RESULTS for query={query!r}: [Line3D, _verts3d]"


@observe_a2a_receive(
    source_agent_id="Planner",
    target_agent_id="Coder",
    edge_id="Planner->Coder",
    message_id="msg-0001",
    channel="requirements",
    message_body_fn=lambda env: env.body,
    carrier_fn=lambda env: env.headers,  # <-- extract trace context from headers
)
def on_message(env: Envelope) -> None:
    # Simulate a tool call triggered by received message.
    _ = tool_code_search("find Line3D _verts3d")


@observe_agent_step(agent_id="Planner", step_index=0)
def planner_step() -> None:
    send_message(Envelope(message_id="msg-0001", body="Implement feature X and write tests."))


@observe_agent_step(agent_id="Coder", step_index=0)
def coder_step() -> None:
    env = INBOX.pop(0)
    on_message(env)


@observe_session(session_id="session-0001")
def run() -> None:
    with segment(name="planning", order=0):
        planner_step()
        coder_step()


if __name__ == "__main__":
    init_otlp_tracing(service_name="llmmas-otel-m1", endpoint="http://localhost:4317", insecure=True)
    run()
