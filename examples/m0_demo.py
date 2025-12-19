from llmmas_otel import init_console_tracing, observe_agent_step, observe_a2a_send, observe_session


@observe_a2a_send(
    source_agent_id="Planner",
    target_agent_id="Coder",
    edge_id="Planner->Coder",
    message_id="msg-0001",
    channel="requirements",
)
def send_message(payload: str) -> None:
    # Milestone 0: fake for now :)
    print("SEND:", payload)


@observe_agent_step(agent_id="Planner", step_index=0)
def planner_step() -> None:
    send_message("Implement feature X and write tests.")


@observe_session(session_id="session-0001")
def run() -> None:
    planner_step()


if __name__ == "__main__":
    init_console_tracing()
    run()
