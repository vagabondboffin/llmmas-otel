from llmmas_otel import init_otlp_tracing, observe_agent_step, observe_a2a_send, observe_session
from llmmas_otel import enable_message_store
enable_message_store("out/messages.jsonl")

from llmmas_otel import segment

@observe_a2a_send(
    source_agent_id="Planner",
    target_agent_id="Coder",
    edge_id="Planner->Coder",
    message_id="msg-0001",
    channel="requirements",
    message_body_fn=lambda payload: payload,  # <-- extract body from send_message(payload)
)
def send_message(payload: str) -> None:
    # Milestone 0: fake for now :)
    print("SEND:", payload)


@observe_agent_step(agent_id="Planner", step_index=0)
def planner_step() -> None:
    send_message("Implement feature X and write tests.")


@observe_session(session_id="session-0001")
def run() -> None:
    with segment(name="planning", index=0):
        planner_step()


if __name__ == "__main__":
    # init_console_tracing(service_name="llmmas_otel-m0")
    init_otlp_tracing(service_name="llmmas-otel-m0", endpoint="http://localhost:4317", insecure=True)

    run()
