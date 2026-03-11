from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import urllib.request

from llmmas_otel.bootstrap import init_otlp_tracing
from llmmas_otel.message_store import enable_message_store
from llmmas_otel.span_factory import default_span_factory
from llmmas_otel.injection.exceptions import (
    LLMFaultError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMNetworkError,
)


# -----------------------------
# Data model (simple A2A envelope)
# -----------------------------
@dataclass
class Envelope:
    message_id: str
    body: str
    channel: str = "requirements"
    headers: dict[str, str] = field(default_factory=dict)


INBOX: list[Envelope] = []


# -----------------------------
# Dataset loading (ProgramDev)
# -----------------------------
def _candidate_dataset_paths() -> list[Path]:
    """
    Default dataset lookup locations.
    Priority:
      1) examples/programdev_dataset.json (your chosen simple path)
      2) ChatDev submodule dataset
      3) sample dataset under demos
      4) repo root programdev_dataset.json
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # .../llmmas-otel
    return [
        repo_root / "examples" / "programdev_dataset.json",
        repo_root / "demos" / "chatdev-ollama" / "ChatDev-Ollama" / "programdev_dataset.json",
        repo_root / "demos" / "chatdev-ollama" / "data" / "programdev_sample3.json",
        repo_root / "programdev_dataset.json",
    ]


def load_tasks(dataset_path: str | None, limit: int) -> list[dict[str, Any]]:
    if dataset_path:
        p = Path(dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    else:
        p = None
        for cand in _candidate_dataset_paths():
            if cand.exists():
                p = cand
                break
        if p is None:
            raise FileNotFoundError(
                "Could not find ProgramDev dataset. Pass --dataset PATH, or place it under "
                "demos/chatdev-ollama/ChatDev-Ollama/programdev_dataset.json"
            )

    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("ProgramDev dataset must be a JSON list of tasks")
    return data[:limit]


# -----------------------------
# Ollama OpenAI-compatible client (no extra deps)
# -----------------------------
def _ollama_base_url() -> str:
    # either OPENAI_BASE_URL (OpenAI-compat) or OLLAMA_BASE_URL; default is local Ollama
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    # normalize: remove trailing /v1 if present (we’ll add it)
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def ollama_chat_completion(messages: list[dict[str, str]], *, model: str, timeout_s: int = 120) -> str:
    """
    Calls Ollama via OpenAI-compatible API: POST {base}/v1/chat/completions
    Returns assistant content string.
    """
    url = f"{_ollama_base_url()}/v1/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": 0.0,
        "max_tokens": 1024,
    }

    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    req = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8")
        data = json.loads(raw)

    # OpenAI-style response
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # be robust across slight variants
        if isinstance(data, dict) and "response" in data:
            return str(data["response"])
        return json.dumps(data)[:500]


def llm_call_with_retry(messages: list[dict[str, str]], *, model: str, max_attempts: int = 3) -> str:
    """
    Minimal retry loop so injected llm.rate_limit causes multiple LLM_CALL spans.
    We retry only on injected LLMFaultError subclasses (rate limit / timeout / network).
    """
    backoff_s = 0.25
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            # LLM_CALL span + injection is handled here
            input_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            with default_span_factory.llm_call(
                provider_name="ollama",
                model=model,
                input_text=input_text,
                record_input=True,
            ) as ctx:
                # enforce injected decisions (RAISE/RETURN handled by decorator normally, but here we do it explicitly)
                dec = default_span_factory.current_llm_call_decision()
                if dec is not None:
                    # DecisionKind is in llmmas_otel.injection, but we avoid importing it here.
                    kind = getattr(getattr(dec, "kind", None), "value", None)
                    if kind == "raise":
                        exc = getattr(dec, "raise_exception", None) or RuntimeError("Injected LLM error")
                        # record exception on span
                        try:
                            from opentelemetry.trace.status import Status, StatusCode
                            ctx.span.record_exception(exc)
                            ctx.span.set_status(Status(StatusCode.ERROR, str(exc)))
                        except Exception:
                            pass
                        raise exc
                    if kind == "return":
                        return str(getattr(dec, "return_value", ""))

                out = ollama_chat_completion(messages, model=model)
                # record output lightly
                try:
                    from llmmas_otel import semconv
                    import hashlib
                    ctx.span.set_attribute(semconv.ATTR_LLM_OUTPUT_PREVIEW, out[:200])
                    ctx.span.set_attribute(semconv.ATTR_LLM_OUTPUT_SHA256, hashlib.sha256(out.encode("utf-8")).hexdigest())
                except Exception:
                    pass
                return out

        except (LLMFaultError, LLMRateLimitError, LLMTimeoutError, LLMNetworkError) as e:
            last_exc = e
            if attempt == max_attempts:
                raise
            time.sleep(backoff_s)
            backoff_s *= 2.0

    # should never reach
    raise RuntimeError(f"LLM call failed after retries: {last_exc}")


# -----------------------------
# Mini MAS: Planner -> Coder
# -----------------------------
def send_message(env: Envelope) -> None:
    with default_span_factory.a2a_send(
        source_agent_id="Planner",
        target_agent_id="Coder",
        edge_id="Planner->Coder",
        message_id=env.message_id,
        channel=env.channel,
        message_body=env.body,
        carrier=env.headers,
        propagate_context=True,
        apply_mutation=lambda new_body: setattr(env, "body", new_body),
    ) as ctx:
        # enforce DROP (skip deliver)
        kind = getattr(getattr(ctx.decision, "kind", None), "value", None)
        if kind == "drop":
            return

        INBOX.append(env)
        print("SEND:", env.body)


def on_message(env: Envelope) -> str | None:
    with default_span_factory.a2a_receive(
        source_agent_id="Planner",
        target_agent_id="Coder",
        edge_id="Planner->Coder",
        message_id=env.message_id,
        channel=env.channel,
        message_body=env.body,
        carrier=env.headers,
        link_from_carrier=True,
    ):
        # enforce DROP on receive
        dec = default_span_factory.current_a2a_receive_decision()
        if getattr(getattr(dec, "kind", None), "value", None) == "drop":
            return None

        return env.body


def planner_step(task: dict[str, Any], *, model: str) -> None:
    with default_span_factory.agent_step(agent_id="Planner", step_index=0):
        prompt = task.get("description") or task.get("task_prompt") or json.dumps(task)[:500]

        messages = [
            {"role": "system", "content": "You are a software planner. Produce a short implementation plan and test plan."},
            {"role": "user", "content": prompt},
        ]
        plan = llm_call_with_retry(messages, model=model, max_attempts=3)

        send_message(Envelope(message_id="msg-0001", body=plan, channel="requirements"))


def coder_step(task: dict[str, Any], *, model: str) -> str:
    with default_span_factory.agent_step(agent_id="Coder", step_index=0):
        env = INBOX.pop(0)
        plan = on_message(env)
        if plan is None:
            return "Coder received DROPPED message."

        prompt = task.get("description") or task.get("task_prompt") or ""
        messages = [
            {"role": "system", "content": "You are a software engineer. Produce a concise final answer: key files + functions + tests. No extra prose."},
            {"role": "user", "content": f"Task:\n{prompt}\n\nPlan from planner:\n{plan}"},
        ]
        out = llm_call_with_retry(messages, model=model, max_attempts=3)
        return out


def run_one(task: dict[str, Any], *, model: str, task_index: int) -> str:
    project = task.get("project_name") or task.get("name") or f"task-{task_index}"
    session_id = f"programdev-demo::{project}"

    with default_span_factory.session(session_id=session_id):
        # Two SE-ish phases
        with default_span_factory.segment(name="planning", order=0):
            planner_step(task, model=model)
        with default_span_factory.segment(name="coding", order=1):
            result = coder_step(task, model=model)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Path to programdev_dataset.json")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--model", type=str, default=os.getenv("OLLAMA_MODEL", "llama2"))
    parser.add_argument("--otlp", type=str, default="http://localhost:4317")
    args = parser.parse_args()

    # tracing + message store
    init_otlp_tracing(service_name="llmmas-demo-programdev", endpoint=args.otlp, insecure=True)
    Path("out").mkdir(exist_ok=True)
    enable_message_store("out/messages_demo.jsonl")

    tasks = load_tasks(args.dataset, args.limit)

    for i, t in enumerate(tasks):
        INBOX.clear()
        print("\n=== TASK", i, "===")
        result = run_one(t, model=args.model, task_index=i)
        print("RESULT (Coder):")
        print(result[:800])


if __name__ == "__main__":
    main()