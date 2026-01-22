import json
import argparse
from pathlib import Path

from chatdev.chat_chain import ChatChain
from camel.typing import ModelType

from llmmas_otel.bootstrap import init_otlp_tracing
from llmmas_otel.message_store import enable_message_store
from llmmas_otel.span_factory import default_span_factory


def get_config_paths(company: str):
    root = Path(__file__).resolve().parent
    config_dir = root / "CompanyConfig" / company
    default_dir = root / "CompanyConfig" / "Default"
    files = ["ChatChainConfig.json", "PhaseConfig.json", "RoleConfig.json"]
    paths = []
    for f in files:
        p = config_dir / f
        paths.append(str(p if p.exists() else (default_dir / f)))
    return tuple(paths)


def run_one_task(task, config_name, org_name, model_type):
    project_name = task["project_name"]
    prompt = task["description"]

    session_id = f"programdev::{project_name}"

    with default_span_factory.session(session_id=session_id):
        config_path, phase_path, role_path = get_config_paths(config_name)

        chain = ChatChain(
            config_path=config_path,
            config_phase_path=phase_path,
            config_role_path=role_path,
            task_prompt=prompt,
            project_name=project_name,
            org_name=org_name,
            model_type=model_type,
            code_path=""
        )

        chain.pre_processing()
        chain.make_recruitment()
        chain.execute_chain()
        chain.post_processing()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="programdev_dataset.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--config", type=str, default="Default")
    parser.add_argument("--org", type=str, default="ProgramDevOrg")
    args = parser.parse_args()

    # OTel exporter -> Jaeger OTLP (you must run Jaeger container)
    init_otlp_tracing(service_name="chatdev-programdev", endpoint="http://localhost:4317", insecure=True)

    # Store full messages to JSONL for later fault injection comparisons
    enable_message_store("out/messages.jsonl")

    tasks = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    tasks = tasks[: args.limit]

    for t in tasks:
        run_one_task(t, args.config, args.org, ModelType.GPT_3_5_TURBO)


if __name__ == "__main__":
    main()
