from __future__ import annotations

import functools
import hashlib
import inspect
import os
import re
import uuid
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from typing import Any, Callable, Iterable, Mapping, Optional

from opentelemetry.trace.status import Status, StatusCode

from llmmas_otel import semconv
from llmmas_otel.message_store import enable_message_store
from llmmas_otel.span_factory import default_span_factory


# ---------------------------------------------------------------------------
# Adapter state
# ---------------------------------------------------------------------------

_PATCHED: set[str] = set()

_CURRENT_HYPERAGENT_SESSION_ID: ContextVar[Optional[str]] = ContextVar(
    "llmmas_hyperagent_session_id",
    default=None,
)

_CURRENT_AGENT_NAME: ContextVar[Optional[str]] = ContextVar(
    "llmmas_hyperagent_agent_name",
    default=None,
)

_STEP_COUNTERS: defaultdict[tuple[str, str], int] = defaultdict(int)


OUTER_CHILD_AGENTS = {"Navigator", "Editor", "Executor"}
INNER_CHAT_MANAGERS = {
    "Navigator": "Navigator Manager",
    "Editor": "Editor Manager",
    "Executor": "Executor Manager",
}
INNER_AGENT_PARENT = {
    "Inner-Navigator-Assistant": "Navigator",
    "Navigator Interpreter": "Navigator",
    "Inner-Editor-Assistant": "Editor",
    "Editor Interpreter": "Editor",
    "Inner-Executor-Assistant": "Executor",
    "Executor Interpreter": "Executor",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def instrument_hyperagent(
    *,
    enable_messages: bool = True,
    message_store_path: Optional[str] = None,
    patch_autogen: bool = True,
    patch_hyperagent_tools: bool = True,
    patch_executors: bool = True,
    patch_llm_calls: bool = True,
) -> None:
    """
    Runtime instrumentation adapter for HyperAgent.

    This function is intentionally HyperAgent-specific, but it maps HyperAgent
    concepts onto MAS-agnostic llmmas-otel concepts:

    - HyperAgent.query_codebase(...) -> session + outer workflow
    - AutoGen agent reply -> agent_step
    - Planner choosing Navigator/Editor/Executor -> delegation + routed message
    - SocietyOfMind child calls -> nested workflow / inner_chat
    - Jupyter/Docker execution -> environment_action
    - HyperAgent BaseTool._run(...) -> environment_action(kind="tool")
    - AutoGen OpenAIWrapper.create(...) -> llm_call, when available

    The core llmmas-otel model remains MAS-agnostic.
    """
    if enable_messages:
        path = (
            message_store_path
            or os.environ.get("LLMMAS_OTEL_MESSAGE_STORE")
            or os.environ.get("HYPERAGENT_OTEL_MESSAGE_STORE")
            or "outputs/llmmas_otel/hyperagent_messages.jsonl"
        )
        enable_message_store(path)

    _patch_hyperagent_query_codebase()

    if patch_autogen:
        _patch_autogen_agent_replies()

    if patch_executors:
        _patch_code_executors()

    if patch_hyperagent_tools:
        _patch_hyperagent_tool_runs()

    if patch_llm_calls:
        _patch_autogen_llm_calls()


# Alias with a shorter name if you prefer.
install = instrument_hyperagent


# ---------------------------------------------------------------------------
# Layer 1: HyperAgent run/session boundary
# ---------------------------------------------------------------------------

def _patch_hyperagent_query_codebase() -> None:
    try:
        from hyperagent.pilot import HyperAgent
    except Exception:
        return

    key = "hyperagent.pilot.HyperAgent.query_codebase"
    if key in _PATCHED:
        return

    original = HyperAgent.query_codebase

    @functools.wraps(original)
    def wrapped(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
        session_id = _make_session_id(self, query)
        token = _CURRENT_HYPERAGENT_SESSION_ID.set(session_id)

        metadata = {
            "repo_path": getattr(self, "repo_path", None),
            "repo_dir": getattr(self, "repo_dir", None),
            "language": getattr(self, "language", None),
            "verbose": getattr(self, "verbose", None),
            "query_sha256": _sha256(query),
        }

        try:
            with default_span_factory.session(
                session_id=session_id,
                name="HyperAgent query_codebase",
                task_id=_infer_task_id(self, query),
                framework="autogen",
                system="hyperagent",
                adapter="llmmas_otel.integrations.hyperagent",
                metadata=metadata,
            ):
                with default_span_factory.workflow(
                    name="HyperAgent outer group chat",
                    kind="outer_chat",
                    order=0,
                    origin="HyperAgent.query_codebase",
                    metadata={
                        "manager": "GroupChatManager",
                        "manager_name": "hyperagent",
                    },
                ):
                    return original(self, query, *args, **kwargs)
        finally:
            _CURRENT_HYPERAGENT_SESSION_ID.reset(token)

    HyperAgent.query_codebase = wrapped
    _PATCHED.add(key)


# ---------------------------------------------------------------------------
# Layer 2: AutoGen agent turns, routed messages, and Planner delegation
# ---------------------------------------------------------------------------

def _patch_autogen_agent_replies() -> None:
    try:
        from autogen import ConversableAgent
    except Exception:
        return

    _patch_class_method_once(
        ConversableAgent,
        "generate_reply",
        _make_generate_reply_wrapper,
        patch_key="autogen.ConversableAgent.generate_reply",
        only_if_defined_on_class=False,
    )

    try:
        from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent

        # Patch only if SocietyOfMindAgent overrides generate_reply itself.
        # If it inherits from ConversableAgent, the parent patch already applies.
        _patch_class_method_once(
            SocietyOfMindAgent,
            "generate_reply",
            _make_generate_reply_wrapper,
            patch_key="autogen.SocietyOfMindAgent.generate_reply",
            only_if_defined_on_class=True,
        )
    except Exception:
        pass


def _make_generate_reply_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        agent_name = _agent_name(self)
        source_agent = _infer_source_agent_from_generate_reply(args, kwargs)
        message_body = _infer_last_message_content(args, kwargs)
        message_id = _make_message_id(source_agent, agent_name, message_body)

        contexts = []

        if source_agent and source_agent != agent_name:
            contexts.append(
                default_span_factory.a2a_receive(
                    source_agent_id=source_agent,
                    target_agent_id=agent_name,
                    edge_id=f"{source_agent}->{agent_name}",
                    message_id=message_id,
                    channel="autogen",
                    message_body=message_body,
                    route_via="AutoGen GroupChatManager",
                    message_kind=_infer_message_kind(message_body),
                )
            )

        inner_parent = INNER_AGENT_PARENT.get(agent_name)
        if agent_name in OUTER_CHILD_AGENTS:
            contexts.append(
                default_span_factory.workflow(
                    name=f"{agent_name} SocietyOfMind call",
                    kind="delegated_agent",
                    origin="AutoGen SocietyOfMindAgent.generate_reply",
                    metadata={
                        "agent": agent_name,
                        "manager": INNER_CHAT_MANAGERS.get(agent_name),
                    },
                )
            )
        elif inner_parent is not None:
            contexts.append(
                default_span_factory.workflow(
                    name=f"{inner_parent} inner chat",
                    kind="inner_chat",
                    origin="AutoGen inner group chat",
                    metadata={
                        "parent_agent": inner_parent,
                        "inner_agent": agent_name,
                    },
                )
            )

        step_index = _next_step_index(agent_name)
        contexts.append(
            default_span_factory.agent_step(
                agent_id=agent_name,
                step_index=step_index,
                agent_role=_infer_agent_role(agent_name),
                agent_impl=type(self).__name__,
                parent_agent_id=inner_parent,
                step_kind="reply",
            )
        )

        token = _CURRENT_AGENT_NAME.set(agent_name)
        result = None

        try:
            with _enter_all(contexts):
                result = original(self, *args, **kwargs)
                _record_agent_reply(agent_name, result)

                if agent_name == "Planner":
                    _record_planner_delegation(result)

                return result

        except Exception as exc:
            _record_exception_on_current_span_if_possible(exc)
            raise

        finally:
            _CURRENT_AGENT_NAME.reset(token)

    return wrapped


def _record_agent_reply(agent_name: str, result: Any) -> None:
    content = _content_from_reply(result)
    if content is None:
        return

    target = _infer_reply_target(agent_name, content)
    if target is None:
        return

    message_id = _make_message_id(agent_name, target, content)

    with default_span_factory.a2a_send(
        source_agent_id=agent_name,
        target_agent_id=target,
        edge_id=f"{agent_name}->{target}",
        message_id=message_id,
        channel="autogen",
        message_body=content,
        route_via="AutoGen GroupChatManager",
        message_kind=_infer_message_kind(content),
        propagate_context=False,
    ):
        pass


def _record_planner_delegation(result: Any) -> None:
    content = _content_from_reply(result)
    if not content:
        return

    delegated_to = _infer_planner_delegate(content)
    if delegated_to is None:
        return

    delegation_id = f"delegation-{uuid.uuid4().hex[:12]}"

    with default_span_factory.delegation(
        from_agent_id="Planner",
        to_agent_id=delegated_to,
        delegation_id=delegation_id,
        kind="subtask",
        via="AutoGen GroupChatManager",
        goal=content,
        metadata={
            "selection_rule": "Planner reply mentions child agent name",
        },
    ):
        pass


# ---------------------------------------------------------------------------
# Layer 3: execution environments
# ---------------------------------------------------------------------------

def _patch_code_executors() -> None:
    # HyperAgent subclasses
    try:
        from hyperagent import build

        for cls_name in ("EICE", "DCLCE"):
            cls = getattr(build, cls_name, None)
            if cls is not None:
                _patch_class_method_once(
                    cls,
                    "execute_code_blocks",
                    _make_execute_code_blocks_wrapper,
                    patch_key=f"hyperagent.build.{cls_name}.execute_code_blocks",
                    only_if_defined_on_class=False,
                )
    except Exception:
        pass

    # AutoGen base classes, as fallback.
    try:
        from autogen.coding.jupyter import EmbeddedIPythonCodeExecutor

        _patch_class_method_once(
            EmbeddedIPythonCodeExecutor,
            "execute_code_blocks",
            _make_execute_code_blocks_wrapper,
            patch_key="autogen.EmbeddedIPythonCodeExecutor.execute_code_blocks",
            only_if_defined_on_class=False,
        )
    except Exception:
        pass

    try:
        from autogen.coding import DockerCommandLineCodeExecutor

        _patch_class_method_once(
            DockerCommandLineCodeExecutor,
            "execute_code_blocks",
            _make_execute_code_blocks_wrapper,
            patch_key="autogen.DockerCommandLineCodeExecutor.execute_code_blocks",
            only_if_defined_on_class=False,
        )
    except Exception:
        pass


def _make_execute_code_blocks_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self: Any, code_blocks: Any, *args: Any, **kwargs: Any) -> Any:
        code_text = _code_blocks_to_text(code_blocks)
        executor_kind = _infer_executor_kind(self)
        action_name = f"{executor_kind}.execute_code_blocks"

        with default_span_factory.environment_action(
            name=action_name,
            kind=executor_kind,
            input_text=code_text,
            record_input=True,
            metadata={
                "executor_class": type(self).__name__,
                "num_code_blocks": len(code_blocks) if hasattr(code_blocks, "__len__") else None,
                "current_agent": _CURRENT_AGENT_NAME.get(),
            },
        ) as ctx:
            try:
                result = original(self, code_blocks, *args, **kwargs)
                _annotate_execution_result(ctx.span, result)
                return result
            except Exception as exc:
                ctx.span.record_exception(exc)
                ctx.span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    return wrapped


# ---------------------------------------------------------------------------
# Layer 4: HyperAgent tools
# ---------------------------------------------------------------------------

def _patch_hyperagent_tool_runs() -> None:
    module_names = [
        "hyperagent.tools.tools",
        "hyperagent.tools.gen_tools",
        "hyperagent.tools.nav_tools",
    ]

    for module_name in module_names:
        try:
            module = __import__(module_name, fromlist=["*"])
        except Exception:
            continue

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if hasattr(obj, "_run"):
                _patch_class_method_once(
                    obj,
                    "_run",
                    _make_tool_run_wrapper,
                    patch_key=f"{module_name}.{obj.__name__}._run",
                    only_if_defined_on_class=True,
                )


def _make_tool_run_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        tool_name = getattr(self, "name", None) or type(self).__name__
        input_text = _tool_args_to_text(args, kwargs)

        with default_span_factory.environment_action(
            name=str(tool_name),
            kind="tool",
            input_text=input_text,
            record_input=True,
            tool_type=type(self).__name__,
            metadata={
                "tool_class": type(self).__name__,
                "current_agent": _CURRENT_AGENT_NAME.get(),
            },
        ) as ctx:
            try:
                result = original(self, *args, **kwargs)
                result_text = _safe_str(result)
                ctx.span.set_attribute(
                    semconv.ATTR_ENV_ACTION_OUTPUT_PREVIEW,
                    result_text[:500],
                )
                ctx.span.set_attribute(
                    semconv.ATTR_ENV_ACTION_OUTPUT_SHA256,
                    _sha256(result_text),
                )
                return result
            except Exception as exc:
                ctx.span.record_exception(exc)
                ctx.span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    return wrapped


# ---------------------------------------------------------------------------
# Layer 5: AutoGen LLM calls, when available
# ---------------------------------------------------------------------------

def _patch_autogen_llm_calls() -> None:
    possible_targets = [
        ("autogen.oai.client", "OpenAIWrapper", "create"),
        ("autogen.oai.client", "OpenAIClient", "create"),
        ("autogen.oai.client", "AzureOpenAIClient", "create"),
    ]

    for module_name, class_name, method_name in possible_targets:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except Exception:
            continue

        _patch_class_method_once(
            cls,
            method_name,
            _make_llm_create_wrapper,
            patch_key=f"{module_name}.{class_name}.{method_name}",
            only_if_defined_on_class=False,
        )


def _make_llm_create_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        model = _infer_llm_model(args, kwargs, self)
        provider = _infer_llm_provider(args, kwargs, self)
        input_text = _infer_llm_input(args, kwargs)

        with default_span_factory.llm_call(
            provider_name=provider,
            model=model,
            operation_name="inference",
            input_text=input_text,
            record_input=True,
            agent_id=_CURRENT_AGENT_NAME.get(),
            metadata={
                "client_class": type(self).__name__,
            },
        ) as ctx:
            try:
                result = original(self, *args, **kwargs)
                output_text = _infer_llm_output(result)
                if output_text is not None:
                    ctx.span.set_attribute(semconv.ATTR_LLM_OUTPUT_PREVIEW, output_text[:500])
                    ctx.span.set_attribute(semconv.ATTR_LLM_OUTPUT_SHA256, _sha256(output_text))
                return result
            except Exception as exc:
                ctx.span.record_exception(exc)
                ctx.span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    return wrapped


# ---------------------------------------------------------------------------
# Optional artifact helper for the next main.py step
# ---------------------------------------------------------------------------

def record_patch_artifact(
    *,
    patch: str,
    instance_id: Optional[str] = None,
    path: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Record a patch as an artifact/state-delta.

    We will call this from HyperAgent's main.py after task.run(...) returns the
    patch and writes it to outputs/runs/<run_name>/patches/<instance_id>.patch.
    """
    name = f"{instance_id}.patch" if instance_id else "patch"
    with default_span_factory.artifact(
        kind="patch",
        name=name,
        path=path,
        content=patch,
        metadata={
            "instance_id": instance_id,
            "patch_empty": not bool((patch or "").strip()),
            **dict(metadata or {}),
        },
    ):
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_class_method_once(
    cls: type,
    method_name: str,
    wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]],
    *,
    patch_key: str,
    only_if_defined_on_class: bool,
) -> None:
    if patch_key in _PATCHED:
        return

    if only_if_defined_on_class and method_name not in cls.__dict__:
        return

    original = getattr(cls, method_name, None)
    if original is None:
        return

    if getattr(original, "__llmmas_otel_patched__", False):
        _PATCHED.add(patch_key)
        return

    wrapped = wrapper_factory(original)
    setattr(wrapped, "__llmmas_otel_patched__", True)
    setattr(cls, method_name, wrapped)
    _PATCHED.add(patch_key)


@contextmanager
def _enter_all(contexts: Iterable[Any]):
    stack = []
    try:
        for ctx in contexts:
            entered = ctx.__enter__()
            stack.append((ctx, entered))
        yield
    finally:
        while stack:
            ctx, _ = stack.pop()
            ctx.__exit__(None, None, None)


def _make_session_id(hyperagent: Any, query: str) -> str:
    explicit = os.environ.get("HYPERAGENT_OTEL_SESSION_ID")
    if explicit:
        return explicit

    repo_path = getattr(hyperagent, "repo_path", None) or "repo"
    repo_name = str(repo_path).rstrip("/").split("/")[-1]
    task = _infer_task_id(hyperagent, query) or "task"
    return f"hyperagent:{repo_name}:{task}:{uuid.uuid4().hex[:8]}"


def _infer_task_id(hyperagent: Any, query: str) -> Optional[str]:
    env_task = os.environ.get("HYPERAGENT_OTEL_TASK_ID")
    if env_task:
        return env_task

    # SWE-bench prompts usually contain repo/issue details but not always the
    # instance id. Keep this conservative.
    repo_path = getattr(hyperagent, "repo_path", None)
    if repo_path:
        repo_name = str(repo_path).rstrip("/").split("/")[-1]
        if repo_name:
            return repo_name

    digest = _sha256(query or "")[:12]
    return f"query-{digest}"


def _agent_name(agent: Any) -> str:
    return (
        getattr(agent, "name", None)
        or getattr(agent, "_name", None)
        or type(agent).__name__
    )


def _next_step_index(agent_name: str) -> int:
    session_id = _CURRENT_HYPERAGENT_SESSION_ID.get() or "no-session"
    key = (session_id, agent_name)
    idx = _STEP_COUNTERS[key]
    _STEP_COUNTERS[key] += 1
    return idx


def _infer_source_agent_from_generate_reply(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Optional[str]:
    sender = kwargs.get("sender")
    if sender is not None:
        name = _agent_name(sender)
        if name:
            return name

    messages = kwargs.get("messages")
    if messages is None and args:
        # AutoGen commonly passes messages as first positional argument.
        if isinstance(args[0], list):
            messages = args[0]

    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, Mapping):
            return last.get("name") or last.get("role")

    return None


def _infer_last_message_content(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Optional[str]:
    messages = kwargs.get("messages")
    if messages is None and args:
        if isinstance(args[0], list):
            messages = args[0]

    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, Mapping):
            content = last.get("content")
            return _safe_str(content) if content is not None else None

    return None


def _content_from_reply(reply: Any) -> Optional[str]:
    if reply is None:
        return None

    if isinstance(reply, str):
        return reply

    if isinstance(reply, Mapping):
        content = reply.get("content")
        if content is not None:
            return _safe_str(content)

    if isinstance(reply, tuple) and reply:
        for item in reversed(reply):
            content = _content_from_reply(item)
            if content:
                return content

    return _safe_str(reply)


def _infer_reply_target(agent_name: str, content: str) -> Optional[str]:
    if agent_name == "Planner":
        delegated_to = _infer_planner_delegate(content)
        return delegated_to or "Admin"

    if agent_name in OUTER_CHILD_AGENTS:
        return "Planner"

    parent = INNER_AGENT_PARENT.get(agent_name)
    if parent is not None:
        if "Interpreter" in agent_name:
            return f"Inner-{parent}-Assistant"
        return f"{parent} Interpreter"

    if agent_name == "Admin":
        return "Planner"

    return None


def _infer_planner_delegate(content: str) -> Optional[str]:
    if not content:
        return None

    # HyperAgent's speaker selector checks these substrings in Planner messages.
    for child in ("Navigator", "Editor", "Executor"):
        if child in content:
            return child

    return None


def _infer_message_kind(content: Optional[str]) -> str:
    if not content:
        return "message"

    lowered = content.lower()
    if "final answer" in lowered:
        return "final_answer"
    if "subgoal" in lowered or "request" in lowered:
        return "instruction"
    if "observation" in lowered:
        return "observation"
    if "error" in lowered or "traceback" in lowered:
        return "error"
    if "_run(" in content or "```bash" in content or "```python" in content:
        return "tool_request"

    return "message"


def _infer_agent_role(agent_name: str) -> Optional[str]:
    mapping = {
        "Admin": "user_proxy",
        "Planner": "planner",
        "Navigator": "navigator",
        "Editor": "editor",
        "Executor": "executor",
        "Inner-Navigator-Assistant": "inner_assistant",
        "Navigator Interpreter": "interpreter",
        "Inner-Editor-Assistant": "inner_assistant",
        "Editor Interpreter": "interpreter",
        "Inner-Executor-Assistant": "inner_assistant",
        "Executor Interpreter": "interpreter",
    }
    return mapping.get(agent_name)


def _make_message_id(source: Optional[str], target: Optional[str], body: Optional[str]) -> str:
    base = f"{source or 'unknown'}->{target or 'unknown'}:{body or ''}"
    return f"msg-{_sha256(base)[:16]}-{uuid.uuid4().hex[:6]}"


def _infer_executor_kind(executor: Any) -> str:
    cls_name = type(executor).__name__.lower()
    if "docker" in cls_name or cls_name == "dclce":
        return "docker_exec"
    if "ipython" in cls_name or "jupyter" in cls_name or cls_name == "eice":
        return "jupyter_exec"
    return "code_exec"


def _code_blocks_to_text(code_blocks: Any) -> str:
    if not code_blocks:
        return ""

    chunks = []
    for block in code_blocks:
        language = getattr(block, "language", None) or ""
        code = getattr(block, "code", None)
        if code is None:
            code = _safe_str(block)
        chunks.append(f"```{language}\n{code}\n```")

    return "\n\n".join(chunks)


def _annotate_execution_result(span: Any, result: Any) -> None:
    exit_code = getattr(result, "exit_code", None)
    output = getattr(result, "output", None)
    code_file = getattr(result, "code_file", None)

    if exit_code is not None:
        span.set_attribute(semconv.ATTR_ENV_ACTION_EXIT_CODE, int(exit_code))
        if int(exit_code) != 0:
            span.set_status(Status(StatusCode.ERROR, f"exit_code={exit_code}"))

    if output is not None:
        output_text = _safe_str(output)
        span.set_attribute(semconv.ATTR_ENV_ACTION_OUTPUT_PREVIEW, output_text[:500])
        span.set_attribute(semconv.ATTR_ENV_ACTION_OUTPUT_SHA256, _sha256(output_text))

    if code_file is not None:
        span.set_attribute("llmmas.env_action.code_file", _safe_str(code_file))


def _tool_args_to_text(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> str:
    parts = []
    if args:
        parts.append("args=" + _safe_str(args))
    if kwargs:
        parts.append("kwargs=" + _safe_str(dict(kwargs)))
    return "\n".join(parts)


def _infer_llm_model(args: tuple[Any, ...], kwargs: Mapping[str, Any], client: Any) -> str:
    for key in ("model", "request_model"):
        if key in kwargs and kwargs[key]:
            return _safe_str(kwargs[key])

    config = kwargs.get("config")
    if isinstance(config, Mapping) and config.get("model"):
        return _safe_str(config["model"])

    for item in args:
        if isinstance(item, Mapping) and item.get("model"):
            return _safe_str(item["model"])

    for attr in ("model", "_model"):
        value = getattr(client, attr, None)
        if value:
            return _safe_str(value)

    return "unknown-model"


def _infer_llm_provider(args: tuple[Any, ...], kwargs: Mapping[str, Any], client: Any) -> str:
    config = kwargs.get("config")
    if isinstance(config, Mapping):
        api_type = config.get("api_type")
        if api_type:
            return _safe_str(api_type)

    for item in args:
        if isinstance(item, Mapping) and item.get("api_type"):
            return _safe_str(item["api_type"])

    cls = type(client).__name__.lower()
    if "azure" in cls:
        return "azure_openai"
    if "openai" in cls:
        return "openai"

    return "unknown-provider"


def _infer_llm_input(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Optional[str]:
    messages = kwargs.get("messages")
    if messages is None:
        for item in args:
            if isinstance(item, Mapping) and "messages" in item:
                messages = item["messages"]
                break
            if isinstance(item, list):
                messages = item
                break

    if messages is None:
        return None

    return _safe_str(messages)


def _infer_llm_output(result: Any) -> Optional[str]:
    if result is None:
        return None

    # OpenAI-style dict response.
    if isinstance(result, Mapping):
        try:
            choices = result.get("choices")
            if choices:
                first = choices[0]
                if isinstance(first, Mapping):
                    message = first.get("message")
                    if isinstance(message, Mapping) and message.get("content") is not None:
                        return _safe_str(message.get("content"))
                    if first.get("text") is not None:
                        return _safe_str(first.get("text"))
        except Exception:
            pass
        return _safe_str(result)

    # Object-style response.
    choices = getattr(result, "choices", None)
    if choices:
        try:
            first = choices[0]
            message = getattr(first, "message", None)
            content = getattr(message, "content", None)
            if content is not None:
                return _safe_str(content)
            text = getattr(first, "text", None)
            if text is not None:
                return _safe_str(text)
        except Exception:
            pass

    return _safe_str(result)


def _record_exception_on_current_span_if_possible(exc: Exception) -> None:
    # The active OpenTelemetry span should already record exceptions in many SDK
    # configurations, but keep this helper for future extension.
    _ = exc


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return repr(value)