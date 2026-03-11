from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple

import urllib.request

# ---- OTel: local trace capture (no Jaeger export needed) ----
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


# -----------------------------
# Minimal A2A envelope
# -----------------------------
@dataclass
class Envelope:
    message_id: str
    body: str
    channel: str = "requirements"
    headers: dict[str, str] = field(default_factory=dict)


INBOX: list[Envelope] = []


# -----------------------------
# Dataset loading
# -----------------------------
def load_tasks(dataset_path: str, limit: int) -> list[dict[str, Any]]:
    p = Path(dataset_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list")
    return data[:limit]


# -----------------------------
# Ollama client (OpenAI-compatible)
# -----------------------------
def _ollama_base_url() -> str:
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


def ollama_chat_completion(messages: list[dict[str, str]], *, model: str, timeout_s: int = 120) -> str:
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

    return data["choices"][0]["message"]["content"]


def llm_call_with_retry(messages: list[dict[str, str]], *, model: str, max_attempts: int = 3) -> str:
    """
    Keep retry logic so Goal B can be tested later.
    For Goal A (delay), retry shouldn't trigger, but it's fine.
    """
    from llmmas_otel.span_factory import default_span_factory
    from llmmas_otel.injection.exceptions import LLMFaultError

    backoff_s = 0.25
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            input_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

            with default_span_factory.llm_call(
                provider_name="ollama",
                model=model,
                input_text=input_text,
                record_input=True,
            ) as ctx:
                dec = default_span_factory.current_llm_call_decision()
                kind = getattr(getattr(dec, "kind", None), "value", None)

                if kind == "raise":
                    exc = getattr(dec, "raise_exception", None) or RuntimeError("Injected LLM error")
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

                # record output lightly (hash+preview)
                try:
                    from llmmas_otel import semconv
                    import hashlib
                    ctx.span.set_attribute(semconv.ATTR_LLM_OUTPUT_PREVIEW, out[:200])
                    ctx.span.set_attribute(semconv.ATTR_LLM_OUTPUT_SHA256, hashlib.sha256(out.encode("utf-8")).hexdigest())
                except Exception:
                    pass

                return out

        except LLMFaultError as e:
            last_exc = e
            if attempt == max_attempts:
                raise
            time.sleep(backoff_s)
            backoff_s *= 2.0

    raise RuntimeError(f"LLM call failed after retries: {last_exc}")


# -----------------------------
# Mini MAS: Planner -> Coder
# -----------------------------
def send_message(env: Envelope) -> None:
    from llmmas_otel.span_factory import default_span_factory

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
        if getattr(getattr(ctx.decision, "kind", None), "value", None) == "drop":
            return
        INBOX.append(env)


def on_message(env: Envelope) -> str | None:
    from llmmas_otel.span_factory import default_span_factory

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
        dec = default_span_factory.current_a2a_receive_decision()
        if getattr(getattr(dec, "kind", None), "value", None) == "drop":
            return None
        return env.body


def planner_step(task: dict[str, Any], *, model: str) -> None:
    from llmmas_otel.span_factory import default_span_factory

    with default_span_factory.agent_step(agent_id="Planner", step_index=0):
        prompt = task.get("description") or task.get("task_prompt") or json.dumps(task)[:500]
        messages = [
            {"role": "system", "content": "You are a software planner. Produce a short implementation plan and test plan."},
            {"role": "user", "content": prompt},
        ]
        plan = llm_call_with_retry(messages, model=model, max_attempts=3)
        send_message(Envelope(message_id="msg-0001", body=plan, channel="requirements"))


def coder_step(task: dict[str, Any], *, model: str) -> str:
    from llmmas_otel.span_factory import default_span_factory

    with default_span_factory.agent_step(agent_id="Coder", step_index=0):
        env = INBOX.pop(0)
        plan = on_message(env)
        if plan is None:
            return "Coder received DROPPED message."

        prompt = task.get("description") or task.get("task_prompt") or ""
        messages = [
            {"role": "system", "content": "You are a software engineer. Produce: key files + functions + tests. Be concise."},
            {"role": "user", "content": f"Task:\n{prompt}\n\nPlan from planner:\n{plan}"},
        ]
        return llm_call_with_retry(messages, model=model, max_attempts=3)


def run_one(task: dict[str, Any], *, model: str, task_index: int) -> str:
    from llmmas_otel.span_factory import default_span_factory

    project = task.get("project_name") or task.get("name") or f"task-{task_index}"
    session_id = f"programdev-demo::{project}"

    with default_span_factory.session(session_id=session_id):
        with default_span_factory.segment(name="planning", order=0):
            planner_step(task, model=model)
        with default_span_factory.segment(name="coding", order=1):
            result = coder_step(task, model=model)

    return result


# -----------------------------
# Trace conversion + indexing
# -----------------------------
def _hex_trace_id(tid: int) -> str:
    return f"{tid:032x}"


def _hex_span_id(sid: int) -> str:
    return f"{sid:016x}"


def spans_to_json(spans) -> dict:
    out = []
    for s in spans:
        ctx = s.get_span_context()
        parent = s.parent.span_id if s.parent is not None else None
        out.append(
            {
                "name": s.name,
                "trace_id": _hex_trace_id(ctx.trace_id),
                "span_id": _hex_span_id(ctx.span_id),
                "parent_span_id": _hex_span_id(parent) if parent is not None else None,
                "kind": str(s.kind.name),
                "start_time_unix_nano": int(s.start_time),
                "end_time_unix_nano": int(s.end_time),
                "attributes": dict(s.attributes) if s.attributes else {},
                "events": [
                    {
                        "name": e.name,
                        "timestamp_unix_nano": int(e.timestamp),
                        "attributes": dict(e.attributes) if e.attributes else {},
                    }
                    for e in (s.events or [])
                ],
                "status": {
                    "status_code": str(getattr(s.status, "status_code", "")),
                    "description": str(getattr(s.status, "description", "")),
                },
            }
        )
    return {"spans": out}


def build_index(spans: list[dict]) -> dict[str, dict]:
    return {s["span_id"]: s for s in spans}


def ancestor_attr(spans_by_id: dict[str, dict], span: dict, key: str) -> Optional[Any]:
    cur = span
    seen = set()
    while cur is not None:
        attrs = cur.get("attributes") or {}
        if key in attrs:
            return attrs[key]
        pid = cur.get("parent_span_id")
        if not pid or pid in seen:
            return None
        seen.add(pid)
        cur = spans_by_id.get(pid)
    return None


def session_id_for_trace(spans: list[dict]) -> Optional[str]:
    for s in spans:
        if s["name"] == "llmmas.session":
            sid = (s.get("attributes") or {}).get("llmmas.session.id")
            return sid if isinstance(sid, str) else None
    return None


def session_span_for_trace(spans: list[dict]) -> Optional[dict]:
    for s in spans:
        if s["name"] == "llmmas.session":
            return s
    return None


def compute_metrics(trace_spans: list[dict]) -> dict:
    sess = session_span_for_trace(trace_spans)
    if sess is None:
        return {}
    session_ms = (sess["end_time_unix_nano"] - sess["start_time_unix_nano"]) / 1e6

    injected_delay_ms = 0.0
    for s in trace_spans:
        for e in s.get("events") or []:
            if e.get("name") == "fault.applied":
                dm = (e.get("attributes") or {}).get("delay_ms")
                if isinstance(dm, (int, float)):
                    injected_delay_ms += float(dm)

    llm_spans = [s for s in trace_spans if (s.get("attributes") or {}).get("gen_ai.operation.name") == "inference"]
    llm_count = len(llm_spans)
    llm_total_ms = sum((s["end_time_unix_nano"] - s["start_time_unix_nano"]) / 1e6 for s in llm_spans)

    return {
        "session_ms": float(session_ms),
        "injected_delay_ms": float(injected_delay_ms),
        "llm_call_count": int(llm_count),
        "llm_total_ms": float(llm_total_ms),
    }


def extract_injection_points(trace_spans: list[dict]) -> list[dict]:
    idx = build_index(trace_spans)
    points = []
    for s in trace_spans:
        attrs = s.get("attributes") or {}
        if attrs.get("llmmas.fault.injected") is not True:
            continue

        delay_ms = None
        for e in s.get("events") or []:
            if e.get("name") == "fault.applied":
                dm = (e.get("attributes") or {}).get("delay_ms")
                if isinstance(dm, (int, float)):
                    delay_ms = float(dm)

        points.append(
            {
                "span_name": s.get("name"),
                "span_kind": s.get("kind"),
                "fault_type": attrs.get("llmmas.fault.type"),
                "fault_spec_id": attrs.get("llmmas.fault.spec_id"),
                "fault_decision": attrs.get("llmmas.fault.decision"),
                "delay_ms": delay_ms,
                "phase_name": ancestor_attr(idx, s, "llmmas.segment.name"),
                "agent_id": ancestor_attr(idx, s, "llmmas.agent.id"),
            }
        )
    return points


def extract_fingerprints(trace_spans: list[dict]) -> dict:
    idx = build_index(trace_spans)

    msg_send: dict[str, str] = {}
    for s in trace_spans:
        attrs = s.get("attributes") or {}
        mid = attrs.get("llmmas.message.id")
        sha = attrs.get("llmmas.message.sha256")
        if isinstance(mid, str) and isinstance(sha, str) and s.get("kind") == "PRODUCER":
            msg_send[mid] = sha

    llm_out: dict[str, list[str]] = {}
    for s in sorted(trace_spans, key=lambda x: x.get("start_time_unix_nano", 0)):
        attrs = s.get("attributes") or {}
        if attrs.get("gen_ai.operation.name") != "inference":
            continue
        out_sha = attrs.get("llmmas.llm.output.sha256")
        if not isinstance(out_sha, str):
            continue
        phase = ancestor_attr(idx, s, "llmmas.segment.name") or "unknown_phase"
        agent = ancestor_attr(idx, s, "llmmas.agent.id") or "unknown_agent"
        k = f"{phase}:{agent}"
        llm_out.setdefault(k, []).append(out_sha)

    return {"a2a_send_sha": msg_send, "llm_output_sha_by_phase_agent": llm_out}


def fingerprints_equal(a: dict, b: dict) -> bool:
    return a == b


# -----------------------------
# Robust stats (median + quartiles)
# -----------------------------
def percentile(sorted_vals: list[float], p: float) -> float:
    """
    Linear interpolation percentile.
    p in [0,1].
    """
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def median_iqr(vals: list[float]) -> dict:
    s = sorted([float(x) for x in vals])
    q1 = percentile(s, 0.25)
    med = percentile(s, 0.50)
    q3 = percentile(s, 0.75)
    return {"median": med, "q1": q1, "q3": q3, "iqr": q3 - q1, "n": len(s)}


# -----------------------------
# Scenario runner
# -----------------------------
def write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def run_once(
    *,
    label: str,
    run_id: str,
    tasks: list[dict[str, Any]],
    model: str,
    exporter: InMemorySpanExporter,
    out_dir: Path,
    faults_yaml_text: Optional[str],
    seed: str,
) -> dict:
    exporter.clear()

    from llmmas_otel import enable_fault_injection, disable_fault_injection, enable_message_store

    disable_fault_injection()

    faults_path = None
    if faults_yaml_text is not None:
        faults_path = out_dir / f"faults_{label}.yaml"
        if not faults_path.exists():
            write_yaml(faults_path, faults_yaml_text)
        enable_fault_injection(str(faults_path), seed=seed)

    enable_message_store(str(out_dir / f"messages_{label}_{run_id}.jsonl"))

    for i, t in enumerate(tasks):
        INBOX.clear()
        _ = run_one(t, model=model, task_index=i)

    trace.get_tracer_provider().force_flush()
    spans = exporter.get_finished_spans()
    spans_json = spans_to_json(spans)

    traces_path = out_dir / f"traces_{label}_{run_id}.json"
    traces_path.write_text(json.dumps(spans_json, indent=2), encoding="utf-8")

    # per-trace grouping -> per-session summary
    by_trace: dict[str, list[dict]] = {}
    for sp in spans_json["spans"]:
        by_trace.setdefault(sp["trace_id"], []).append(sp)

    per_session: dict[str, dict] = {}
    for tid, t_spans in by_trace.items():
        sid = session_id_for_trace(t_spans)
        if sid is None:
            continue
        per_session[sid] = {
            "trace_id": tid,
            "metrics": compute_metrics(t_spans),
            "injection_points": extract_injection_points(t_spans),
            "fingerprints": extract_fingerprints(t_spans),
        }

    return {
        "label": label,
        "run_id": run_id,
        "traces_file": str(traces_path),
        "messages_file": str(out_dir / f"messages_{label}_{run_id}.jsonl"),
        "faults_file": str(faults_path) if faults_path else None,
        "per_session": per_session,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="examples/programdev_dataset.json")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--model", type=str, default=os.getenv("OLLAMA_MODEL", "llama2"))
    parser.add_argument("--delay_ms", type=int, default=1000)
    parser.add_argument("--seed", type=str, default="icst")
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    # tracer provider with in-memory exporter
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "llmmas-demo-eval"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    tasks = load_tasks(args.dataset, args.limit)

    # Scenario YAMLs (separate, controllable)
    a1_yaml = f"""
faults:
  - id: A1_LLM_DELAY
    hook: llm_call
    selector:
      phase_name: planning
    action:
      type: llm.delay
      params:
        delay_ms: {args.delay_ms}
    limits:
      probability: 1.0
      max_times: 1
"""
    a2_yaml = f"""
faults:
  - id: A2_A2A_DELAY
    hook: a2a_send
    selector:
      phase_name: planning
      source_agent_id: Planner
      target_agent_id: Coder
    action:
      type: a2a.delay
      params:
        delay_ms: {args.delay_ms}
    limits:
      probability: 1.0
      max_times: 1
"""

    scenarios = {
        "baseline": None,
        "a1_llm_delay": a1_yaml,
        "a2_a2a_delay": a2_yaml,
    }

    # Run repeated trials
    runs: dict[str, list[dict]] = {k: [] for k in scenarios}
    for r in range(1, args.repeats + 1):
        run_id = f"r{r:02d}"
        for label, yml in scenarios.items():
            runs[label].append(
                run_once(
                    label=label,
                    run_id=run_id,
                    tasks=tasks,
                    model=args.model,
                    exporter=exporter,
                    out_dir=out_dir,
                    faults_yaml_text=yml,
                    seed=args.seed,
                )
            )
        print(f"Completed repeat {r}/{args.repeats}")

    # Aggregate by session_id
    all_session_ids = set()
    for label in runs:
        for run in runs[label]:
            all_session_ids |= set(run["per_session"].keys())

    # Baseline fingerprints for content equality check
    baseline_fps: dict[str, list[dict]] = {}
    for run in runs["baseline"]:
        for sid, entry in run["per_session"].items():
            baseline_fps.setdefault(sid, []).append(entry["fingerprints"])

    report: dict[str, Any] = {
        "meta": {
            "dataset": args.dataset,
            "limit": args.limit,
            "model": args.model,
            "delay_ms": args.delay_ms,
            "seed": args.seed,
            "repeats": args.repeats,
            "generated_at_unix": time.time(),
        },
        "scenarios": {
            "baseline": {"faults_yaml": None},
            "a1_llm_delay": {"faults_yaml": a1_yaml, "faults_file": str(out_dir / "faults_a1_llm_delay.yaml")},
            "a2_a2a_delay": {"faults_yaml": a2_yaml, "faults_file": str(out_dir / "faults_a2_a2a_delay.yaml")},
        },
        "artifacts": {
            "runs": {
                label: [{"run_id": x["run_id"], "traces": x["traces_file"], "messages": x["messages_file"]} for x in lst]
                for label, lst in runs.items()
            }
        },
        "results": {},
        "notes": {
            "matching_rule": "Runs are matched by llmmas.session.id (stable). trace_id differs per run.",
            "goal_a_expectation": "Delay faults should increase session_ms without changing content hashes (temp=0).",
        },
    }

    for sid in sorted(all_session_ids):
        report["results"][sid] = {}

        # Collect baseline metrics per repeat for this session_id
        baseline_metrics = []
        baseline_fp_list = baseline_fps.get(sid, [])

        for run in runs["baseline"]:
            entry = run["per_session"].get(sid)
            if entry:
                baseline_metrics.append(entry["metrics"])

        if not baseline_metrics:
            continue

        # helper to pull a list of values
        def vals(metrics_list: list[dict], key: str) -> list[float]:
            out = []
            for m in metrics_list:
                v = m.get(key)
                if isinstance(v, (int, float)):
                    out.append(float(v))
            return out

        base_session_stats = median_iqr(vals(baseline_metrics, "session_ms"))
        base_llm_total_stats = median_iqr(vals(baseline_metrics, "llm_total_ms"))
        base_llm_count_stats = median_iqr(vals(baseline_metrics, "llm_call_count"))

        report["results"][sid]["baseline"] = {
            "session_ms": base_session_stats,
            "llm_total_ms": base_llm_total_stats,
            "llm_call_count": base_llm_count_stats,
        }

        # For each fault scenario, aggregate metrics and compare vs baseline (per-repeat deltas)
        for label in ("a1_llm_delay", "a2_a2a_delay"):
            met_list = []
            injection_points_all = []
            content_changed_flags = []

            for run in runs[label]:
                entry = run["per_session"].get(sid)
                if not entry:
                    continue
                met_list.append(entry["metrics"])
                injection_points_all.extend(entry.get("injection_points") or [])

                # content equality vs baseline fingerprints (use first baseline fp as reference)
                if baseline_fp_list:
                    changed = not fingerprints_equal(baseline_fp_list[0], entry["fingerprints"])
                    content_changed_flags.append(changed)

            if not met_list:
                continue

            session_stats = median_iqr(vals(met_list, "session_ms"))
            inj_delay_stats = median_iqr(vals(met_list, "injected_delay_ms"))
            llm_total_stats = median_iqr(vals(met_list, "llm_total_ms"))
            llm_count_stats = median_iqr(vals(met_list, "llm_call_count"))

            # per-repeat overhead (faulty - baseline) for session_ms
            overhead_vals = []
            amp_vals = []
            for i in range(min(len(met_list), len(baseline_metrics))):
                f = met_list[i]
                b = baseline_metrics[i]
                if not isinstance(f.get("session_ms"), (int, float)) or not isinstance(b.get("session_ms"), (int, float)):
                    continue
                overhead = float(f["session_ms"]) - float(b["session_ms"])
                overhead_vals.append(overhead)

                inj = float(f.get("injected_delay_ms") or 0.0)
                if inj > 0:
                    amp_vals.append(overhead / inj)

            overhead_stats = median_iqr(overhead_vals) if overhead_vals else None
            amp_stats = median_iqr(amp_vals) if amp_vals else None

            # content change rate
            change_rate = None
            if content_changed_flags:
                change_rate = sum(1 for x in content_changed_flags if x) / len(content_changed_flags)

            # choose "representative" injection point (first one) just for readability
            rep_ip = injection_points_all[0] if injection_points_all else None

            report["results"][sid][label] = {
                "session_ms": session_stats,
                "injected_delay_ms": inj_delay_stats,
                "llm_total_ms": llm_total_stats,
                "llm_call_count": llm_count_stats,
                "overhead_ms": overhead_stats,
                "amplification": amp_stats,
                "content_change_rate": change_rate,
                "injection_point_example": rep_ip,
            }

    out_json = out_dir / "report_goal_a_repeats.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nWrote: {out_json}")
    print("Also wrote per-run traces/messages JSON files in out/.")
    print("Tip: open report_goal_a_repeats.json and look under results[session_id].")


if __name__ == "__main__":
    main()