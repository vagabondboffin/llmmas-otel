from __future__ import annotations

import copy
import functools
import hashlib
import inspect
import os
import re
import uuid
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterable, Mapping, Optional

from opentelemetry.trace.status import Status, StatusCode

from llmmas_otel import semconv
from llmmas_otel import message_store
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

# Segment counters are used to create meaningful HyperAgent MAS segments:
# Setup, Planning turn N, Navigation subtask N, Editing subtask N,
# Execution subtask N, Output.
_SEGMENT_COUNTERS: defaultdict[tuple[str, str], int] = defaultdict(int)
_SEGMENT_ORDER_COUNTERS: defaultdict[str, int] = defaultdict(int)

# One pending Planner delegation per active HyperAgent task/session.
_PENDING_DELEGATIONS: dict[str, dict[str, Any]] = {}

OUTER_CHILD_AGENTS = {"Navigator", "Editor", "Executor"}

OUTER_MANAGER_NAMES = {
    "hyperagent",
    "GroupChatManager",
}

INNER_MANAGER_TO_PARENT = {
    "Navigator Manager": "Navigator",
    "Editor Manager": "Editor",
    "Executor Manager": "Executor",
}

INNER_AGENT_PARENT = {
    "Inner-Navigator-Assistant": "Navigator",
    "Navigator Interpreter": "Navigator",
    "Inner-Editor-Assistant": "Editor",
    "Editor Interpreter": "Editor",
    "Inner-Executor-Assistant": "Executor",
    "Executor Interpreter": "Executor",
}

_OPENAI_MESSAGE_NAME_PATTERN = re.compile(r"^[^\s<|\\/>]+$")


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

    This adapter maps HyperAgent/AutoGen boundaries to MAS-agnostic
    llmmas-otel concepts:

    - external task wrapper in main.py -> root session
    - HyperAgent.__init__(...) -> Setup segment
    - HyperAgent.query_codebase(...) -> no visible parent workflow; semantic
      planning/delegation segments are created directly under the task session
    - Planner reply -> Planning turn segment + agent_step
    - Planner selecting Navigator/Editor/Executor -> pending delegation
    - child SocietyOfMindAgent call -> Navigation/Editing/Execution segment
    - child inner manager call -> inner_chat workflow inside the active segment
    - inner assistant/interpreter replies -> agent_step
    - Jupyter/Docker execution and HyperAgent tools -> tool_call operation
    - patch output -> Output segment + artifact, recorded from main.py
    """
    if enable_messages:
        path = (
            message_store_path
            or os.environ.get("LLMMAS_OTEL_MESSAGE_STORE")
            or os.environ.get("HYPERAGENT_OTEL_MESSAGE_STORE")
            or "outputs/llmmas_otel/hyperagent_messages.jsonl"
        )
        enable_message_store(path)

    _patch_hyperagent_init()
    _patch_hyperagent_query_codebase()

    if patch_autogen:
        _patch_autogen_agent_replies()

    if patch_executors:
        _patch_code_executors()

    if patch_hyperagent_tools:
        _patch_hyperagent_tool_runs()

    if patch_llm_calls:
        _patch_autogen_llm_calls()


install = instrument_hyperagent


# ---------------------------------------------------------------------------
# Layer 0: HyperAgent setup boundary
# ---------------------------------------------------------------------------

def _patch_hyperagent_init() -> None:
    try:
        from hyperagent.pilot import HyperAgent
    except Exception:
        return

    key = "hyperagent.pilot.HyperAgent.__init__"
    if key in _PATCHED:
        return

    original = HyperAgent.__init__

    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Setup is part of the same task trace, but it is not performed by
        # Planner/Navigator/Editor/Executor. We still keep the standard
        # hierarchy by representing it as a System agent step.
        if message_store.current_session_id():
            token = _CURRENT_AGENT_NAME.set("System")
            try:
                with _enter_all(_setup_contexts()):
                    return original(self, *args, **kwargs)
            finally:
                _CURRENT_AGENT_NAME.reset(token)

        return original(self, *args, **kwargs)

    HyperAgent.__init__ = wrapped
    _PATCHED.add(key)


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
        existing_session_id = message_store.current_session_id()

        if existing_session_id:
            session_cm = _null_cm()
            session_id = existing_session_id
        else:
            session_id = _make_session_id(self, query)
            session_cm = default_span_factory.session(
                session_id=session_id,
                name=f"Task {_infer_task_id(self, query)}",
                task_id=_infer_task_id(self, query),
                framework="autogen",
                system="hyperagent",
                adapter="llmmas_otel.integrations.hyperagent",
                metadata={
                    "benchmark": "direct_query",
                    "dataset": "direct_query",
                    "repo_path": getattr(self, "repo_path", None),
                    "repo_dir": getattr(self, "repo_dir", None),
                    "language": getattr(self, "language", None),
                    "verbose": getattr(self, "verbose", None),
                    "query_sha256": _sha256(query),
                },
            )

        token = _CURRENT_HYPERAGENT_SESSION_ID.set(session_id)

        try:
            with session_cm:
                return original(self, query, *args, **kwargs)
        finally:
            _CURRENT_HYPERAGENT_SESSION_ID.reset(token)

    HyperAgent.query_codebase = wrapped
    _PATCHED.add(key)


# ---------------------------------------------------------------------------
# Layer 2: AutoGen replies, Planner turns, delegations, inner chats
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

        # AutoGen managers are routing/orchestration mechanisms, not MAS agents.
        # They should not dominate the visible trace hierarchy.
        if _is_outer_manager(agent_name):
            return original(self, *args, **kwargs)

        if agent_name in INNER_MANAGER_TO_PARENT:
            parent_agent = INNER_MANAGER_TO_PARENT[agent_name]
            with default_span_factory.workflow(
                name=f"{parent_agent} inner chat",
                kind="inner_chat",
                origin="AutoGen inner GroupChatManager",
                metadata={
                    "manager": agent_name,
                    "parent_agent": parent_agent,
                    "group_chat.name": "hyperagent",
                    "group_chat.manager": "AutoGen GroupChatManager",
                },
            ):
                return original(self, *args, **kwargs)

        source_agent = _infer_source_agent_for_reply(agent_name, args, kwargs)
        message_body = _infer_last_message_content(args, kwargs)

        # Strict llmmas-otel hierarchy:
        #   session -> segment/phase -> agent_step -> operation
        #
        # Operation spans such as a2a_receive and delegation are short
        # operation spans. They are recorded inside the agent_step and closed
        # before original(...). This keeps later llm_call, a2a_send, and
        # tool_call spans as siblings under the same agent step, which is
        # necessary for fault injection and localization.
        segment_contexts: list[Any] = []
        receive_context: Optional[Any] = None
        delegation_context: Optional[Any] = None

        if agent_name == "Planner":
            segment_contexts.append(_planner_segment_context())
        elif agent_name in OUTER_CHILD_AGENTS:
            segment_contexts.append(_child_segment_context(agent_name))

        pending = _peek_pending_delegation(agent_name)
        if agent_name in OUTER_CHILD_AGENTS and pending is not None:
            source_agent = pending["from_agent"]
            message_body = pending.get("goal") or message_body

        inner_parent = INNER_AGENT_PARENT.get(agent_name)
        step_index = _next_step_index(agent_name)

        agent_step_context = default_span_factory.agent_step(
            agent_id=agent_name,
            step_index=step_index,
            agent_role=_infer_agent_role(agent_name),
            agent_impl=type(self).__name__,
            parent_agent_id=inner_parent,
            step_kind="reply",
            metadata={
                "group_chat.name": "hyperagent",
                "group_chat.manager": "AutoGen GroupChatManager",
            },
        )

        if source_agent and source_agent != agent_name and not _is_manager_name(source_agent):
            receive_context = _a2a_receive_context(
                source_agent_id=source_agent,
                target_agent_id=agent_name,
                edge_id=f"{source_agent}->{agent_name}",
                message_id=_make_message_id(source_agent, agent_name, message_body),
                channel="autogen",
                message_body=message_body,
                route_via="AutoGen GroupChatManager",
                message_kind=_infer_message_kind(message_body),
            )

        if agent_name in OUTER_CHILD_AGENTS:
            delegation_context = _consume_delegation_context(agent_name)

        token = _CURRENT_AGENT_NAME.set(agent_name)

        try:
            with _enter_all([*segment_contexts, agent_step_context]):
                if receive_context is not None:
                    with receive_context:
                        pass

                if delegation_context is not None:
                    with delegation_context:
                        pass

                result = original(self, *args, **kwargs)

                if agent_name == "Planner":
                    _handle_planner_reply(result)
                else:
                    _record_non_planner_reply(agent_name, result)

                return result

        except Exception as exc:
            try:
                from opentelemetry import trace

                span = trace.get_current_span()
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:
                pass
            raise

        finally:
            _CURRENT_AGENT_NAME.reset(token)

    return wrapped


def _handle_planner_reply(result: Any) -> None:
    content = _content_from_reply(result)
    if not content:
        return

    delegated_to = _infer_planner_delegate(content)

    if delegated_to:
        with default_span_factory.a2a_send(
            source_agent_id="Planner",
            target_agent_id=delegated_to,
            edge_id=f"Planner->{delegated_to}",
            message_id=_make_message_id("Planner", delegated_to, content),
            channel="autogen",
            message_body=content,
            route_via="AutoGen GroupChatManager",
            message_kind="instruction",
            propagate_context=False,
        ):
            pass

        _set_pending_delegation(
            from_agent="Planner",
            to_agent=delegated_to,
            goal=content,
        )
    else:
        with default_span_factory.a2a_send(
            source_agent_id="Planner",
            target_agent_id="Admin",
            edge_id="Planner->Admin",
            message_id=_make_message_id("Planner", "Admin", content),
            channel="autogen",
            message_body=content,
            route_via="AutoGen GroupChatManager",
            message_kind=_infer_message_kind(content),
            propagate_context=False,
        ):
            pass


def _record_non_planner_reply(agent_name: str, result: Any) -> None:
    content = _content_from_reply(result)
    if not content:
        return

    if agent_name in OUTER_CHILD_AGENTS:
        target = "Planner"
    elif agent_name in INNER_AGENT_PARENT:
        parent = INNER_AGENT_PARENT[agent_name]
        if "Interpreter" in agent_name:
            target = f"Inner-{parent}-Assistant"
        else:
            target = f"{parent} Interpreter"
    else:
        return

    with default_span_factory.a2a_send(
        source_agent_id=agent_name,
        target_agent_id=target,
        edge_id=f"{agent_name}->{target}",
        message_id=_make_message_id(agent_name, target, content),
        channel="autogen",
        message_body=content,
        route_via="AutoGen GroupChatManager",
        message_kind=_infer_message_kind(content),
        propagate_context=False,
    ):
        pass


# ---------------------------------------------------------------------------
# Pending delegation handling
# ---------------------------------------------------------------------------

def _session_key() -> str:
    return (
        message_store.current_session_id()
        or _CURRENT_HYPERAGENT_SESSION_ID.get()
        or "no-session"
    )


def _set_pending_delegation(*, from_agent: str, to_agent: str, goal: str) -> None:
    _PENDING_DELEGATIONS[_session_key()] = {
        "delegation_id": f"delegation-{uuid.uuid4().hex[:12]}",
        "from_agent": from_agent,
        "to_agent": to_agent,
        "goal": goal,
    }


def _peek_pending_delegation(agent_name: str) -> Optional[dict[str, Any]]:
    pending = _PENDING_DELEGATIONS.get(_session_key())
    if pending and pending.get("to_agent") == agent_name:
        return pending
    return None


def _consume_delegation_context(agent_name: str) -> Optional[Any]:
    pending = _peek_pending_delegation(agent_name)
    if pending is None:
        return None

    _PENDING_DELEGATIONS.pop(_session_key(), None)

    return default_span_factory.delegation(
        from_agent_id=pending["from_agent"],
        to_agent_id=pending["to_agent"],
        delegation_id=pending["delegation_id"],
        kind="subtask",
        via="AutoGen GroupChatManager",
        goal=pending.get("goal"),
        metadata={
            "selection_rule": "Planner reply mentions child agent name",
        },
    )


# ---------------------------------------------------------------------------
# Layer 3: execution environments
# ---------------------------------------------------------------------------

def _patch_code_executors() -> None:
    """
    Patch only HyperAgent executor subclasses.

    We intentionally do not patch AutoGen base executor classes here because
    HyperAgent EICE/DCLCE may call into those base classes. Patching both levels
    creates duplicate nested jupyter_exec/docker_exec spans for one execution.
    """
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


def _make_execute_code_blocks_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self: Any, code_blocks: Any, *args: Any, **kwargs: Any) -> Any:
        code_text = _code_blocks_to_text(code_blocks)
        executor_kind = _infer_executor_kind(self)
        tool_invocations = _extract_tool_invocations_from_code(code_text)
        action_name = _execution_action_name(
            executor_kind=executor_kind,
            code_text=code_text,
            tool_invocations=tool_invocations,
        )

        # HyperAgent interpreter agents use Jupyter/Docker execution as their
        # operation boundary. Expose that boundary as a tool_call under the
        # current agent step, not as a direct environment_action child.
        if _CURRENT_AGENT_NAME.get() is not None:
            return _execute_code_as_tool_operation(
                original=original,
                executor=self,
                code_blocks=code_blocks,
                args=args,
                kwargs=kwargs,
                code_text=code_text,
                executor_kind=executor_kind,
                action_name=action_name,
                tool_invocations=tool_invocations,
            )

        # Fallback for rare execution outside an agent context.
        with default_span_factory.environment_action(
            name=action_name,
            kind=executor_kind,
            input_text=code_text,
            record_input=True,
            metadata={
                "executor_class": type(self).__name__,
                "num_code_blocks": len(code_blocks) if hasattr(code_blocks, "__len__") else None,
                "current_agent": _CURRENT_AGENT_NAME.get(),
                "semantic_action": action_name,
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

        with default_span_factory.tool_call(
            tool_name=str(tool_name),
            tool_call_id=f"toolcall-{uuid.uuid4().hex[:12]}",
            tool_type=type(self).__name__,
            tool_args=input_text,
            record_args=True,
        ) as ctx:
            try:
                ctx.span.update_name(f"llmmas.tool_call {tool_name}")
            except Exception:
                pass

            ctx.span.set_attribute("llmmas.tool.class", type(self).__name__)
            current_agent = _CURRENT_AGENT_NAME.get()
            if current_agent is not None:
                ctx.span.set_attribute("llmmas.agent.id", current_agent)

            try:
                result = original(self, *args, **kwargs)
                result_text = _safe_str(result)
                ctx.span.set_attribute(
                    semconv.ATTR_TOOL_RESULT_PREVIEW,
                    result_text[:500],
                )
                ctx.span.set_attribute(
                    semconv.ATTR_TOOL_RESULT_SHA256,
                    _sha256(result_text),
                )
                return result
            except Exception as exc:
                ctx.span.record_exception(exc)
                ctx.span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise

    return wrapped


def _patch_autogen_llm_calls() -> None:
    """
    Patch AutoGen LLM calls in two layers.

    OpenAIWrapper.create creates the llm_call span. OpenAIClient.create and
    AzureOpenAIClient.create are patched only to sanitize the final params dict
    immediately before the OpenAI-compatible HTTP request. The lower-level
    wrappers do not create spans, so we avoid duplicate LLM spans.
    """
    traced_targets = [
        ("autogen.oai.client", "OpenAIWrapper", "create"),
    ]

    for module_name, class_name, method_name in traced_targets:
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

    sanitizer_targets = [
        ("autogen.oai.client", "OpenAIClient", "create"),
        ("autogen.oai.client", "AzureOpenAIClient", "create"),
    ]

    for module_name, class_name, method_name in sanitizer_targets:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
        except Exception:
            continue

        _patch_class_method_once(
            cls,
            method_name,
            _make_low_level_client_create_sanitizer_wrapper,
            patch_key=f"{module_name}.{class_name}.{method_name}.sanitize_only",
            only_if_defined_on_class=False,
        )


def _make_llm_create_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        sanitized_args, sanitized_kwargs = _sanitize_autogen_openai_create_payload(args, kwargs)

        model = _infer_llm_model(sanitized_args, sanitized_kwargs, self)
        provider = _infer_llm_provider(sanitized_args, sanitized_kwargs, self)
        input_text = _infer_llm_input(sanitized_args, sanitized_kwargs)

        with default_span_factory.llm_call(
            provider_name=provider,
            model=model,
            operation_name="inference",
            input_text=input_text,
            record_input=True,
            agent_id=_CURRENT_AGENT_NAME.get(),
            metadata={
                "client_class": type(self).__name__,
                "openai_message_names_sanitized": True,
            },
        ) as ctx:
            try:
                dec = default_span_factory.current_llm_call_decision()
                if dec is not None:
                    try:
                        from llmmas_otel.injection import DecisionKind
                    except Exception:
                        DecisionKind = None  # type: ignore
                    if DecisionKind is not None and dec.kind == DecisionKind.MUTATE_INPUT:
                        mutator = dec.return_value
                        try:
                            sanitized_args, sanitized_kwargs = mutator(sanitized_args, sanitized_kwargs)
                        except Exception as _e:
                            ctx.span.set_attribute("llmmas.fault.error", f"mutator_failed: {_e}")
                result = original(self, *sanitized_args, **sanitized_kwargs)
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


def _make_low_level_client_create_sanitizer_wrapper(
    original: Callable[..., Any],
) -> Callable[..., Any]:
    """Sanitize AutoGen's final OpenAI/AzureOpenAI params dict."""
    @functools.wraps(original)
    def wrapped(self: Any, params: Any, *args: Any, **kwargs: Any) -> Any:
        sanitized_params = _sanitize_openai_payload_object(params)
        return original(self, sanitized_params, *args, **kwargs)

    return wrapped


def _sanitize_autogen_openai_create_payload(
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """
    Sanitize OpenAI message names before AutoGen sends requests to an
    OpenAI-compatible endpoint.

    OpenAI-compatible APIs reject names with whitespace or characters such as
    < | \\ / >. HyperAgent/AutoGen uses names like "Executor Manager", so direct
    OpenRouter calls fail unless we normalize the API-facing payload.
    """
    new_args = tuple(_sanitize_openai_payload_object(arg) for arg in args)
    new_kwargs = {key: _sanitize_openai_payload_object(value) for key, value in dict(kwargs).items()}
    return new_args, new_kwargs


def _sanitize_openai_payload_object(value: Any) -> Any:
    if isinstance(value, Mapping):
        copied = dict(value)

        if "messages" in copied:
            copied["messages"] = _sanitize_messages(copied["messages"])

        # AutoGen may nest prompt/messages under context/config-like dicts.
        for key, item in list(copied.items()):
            if key != "messages" and isinstance(item, (Mapping, list, tuple)):
                copied[key] = _sanitize_openai_payload_object(item)

        return copied

    if isinstance(value, list):
        return [_sanitize_openai_payload_object(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_sanitize_openai_payload_object(item) for item in value)

    return value


def _sanitize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages

    sanitized = []
    for message in messages:
        if not isinstance(message, Mapping):
            sanitized.append(message)
            continue

        copied = dict(message)
        name = copied.get("name")
        if name is not None:
            safe_name = _sanitize_openai_message_name(str(name))
            if safe_name:
                copied["name"] = safe_name
            else:
                copied.pop("name", None)

        sanitized.append(copied)

    return sanitized


def _sanitize_openai_message_name(name: str) -> str:
    safe = name.strip()
    safe = re.sub(r"[\s<|\\/>]+", "_", safe)
    safe = re.sub(r"_+", "_", safe).strip("_")

    if not safe:
        return ""

    if not _OPENAI_MESSAGE_NAME_PATTERN.match(safe):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", safe)
        safe = re.sub(r"_+", "_", safe).strip("_")

    return safe[:64]


# ---------------------------------------------------------------------------
# Artifact helper called from HyperAgent main.py
# ---------------------------------------------------------------------------

def record_patch_artifact(
    *,
    patch: str,
    instance_id: Optional[str] = None,
    path: Optional[str] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    name = f"{instance_id}.patch" if instance_id else "patch"

    with _output_segment_context():
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
# Semantic segment helpers
# ---------------------------------------------------------------------------

def _next_segment_order() -> int:
    session_id = _session_key()
    value = _SEGMENT_ORDER_COUNTERS[session_id]
    _SEGMENT_ORDER_COUNTERS[session_id] += 1
    return value


def _next_named_segment_index(name: str) -> int:
    session_id = _session_key()
    key = (session_id, name)
    value = _SEGMENT_COUNTERS[key]
    _SEGMENT_COUNTERS[key] += 1
    return value


def _planner_segment_context() -> Any:
    idx = _next_named_segment_index("planning")
    return default_span_factory.segment(
        name=f"Planning turn {idx}",
        order=_next_segment_order(),
        kind="planning",
        origin="HyperAgent Planner",
        metadata={
            "segment.semantic_role": "planner_decision",
            "segment.index": idx,
            "group_chat.name": "hyperagent",
            "group_chat.manager": "AutoGen GroupChatManager",
        },
    )


def _child_segment_context(agent_name: str) -> Any:
    role_to_name = {
        "Navigator": "Navigation subtask",
        "Editor": "Editing subtask",
        "Executor": "Execution subtask",
    }

    role_to_kind = {
        "Navigator": "navigation",
        "Editor": "editing",
        "Executor": "execution",
    }

    base = role_to_name.get(agent_name, f"{agent_name} subtask")
    kind = role_to_kind.get(agent_name, "delegated_subtask")
    idx = _next_named_segment_index(kind)

    return default_span_factory.segment(
        name=f"{base} {idx}",
        order=_next_segment_order(),
        kind=kind,
        origin=f"HyperAgent {agent_name}",
        metadata={
            "segment.semantic_role": kind,
            "segment.index": idx,
            "delegated_agent": agent_name,
            "group_chat.name": "hyperagent",
            "group_chat.manager": "AutoGen GroupChatManager",
        },
    )


def _setup_segment_context() -> Any:
    return default_span_factory.segment(
        name="Setup",
        order=_next_segment_order(),
        kind="setup",
        origin="HyperAgent.__init__",
        metadata={
            "segment.semantic_role": "setup",
        },
    )


def _output_segment_context() -> Any:
    return default_span_factory.segment(
        name="Output",
        order=_next_segment_order(),
        kind="output",
        origin="HyperAgent patch extraction",
        metadata={
            "segment.semantic_role": "output_artifact",
        },
    )


@contextmanager
def _a2a_receive_context(
    *,
    source_agent_id: str,
    target_agent_id: str,
    edge_id: str,
    message_id: str,
    channel: str,
    message_body: Optional[str],
    route_via: str,
    message_kind: str,
):
    with default_span_factory.a2a_receive(
        source_agent_id=source_agent_id,
        target_agent_id=target_agent_id,
        edge_id=edge_id,
        message_id=message_id,
        channel=channel,
        message_body=message_body,
        route_via=route_via,
        message_kind=message_kind,
    ) as span:
        try:
            span.update_name(f"receive {source_agent_id}->{target_agent_id}")
        except Exception:
            pass
        yield span


def _setup_contexts() -> list[Any]:
    return [
        _setup_segment_context(),
        default_span_factory.agent_step(
            agent_id="System",
            step_index=0,
            agent_role="system",
            agent_impl="HyperAgent",
            step_kind="setup",
            metadata={
                "group_chat.name": "hyperagent",
                "setup.kind": "tool_environment",
            },
        ),
    ]


def _execute_code_as_tool_operation(
    *,
    original: Callable[..., Any],
    executor: Any,
    code_blocks: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    code_text: str,
    executor_kind: str,
    action_name: str,
    tool_invocations: list[dict[str, Any]],
) -> Any:
    tool_name = _tool_operation_name(
        executor_kind=executor_kind,
        action_name=action_name,
        tool_invocations=tool_invocations,
    )
    tool_type = _tool_operation_type(
        executor_kind=executor_kind,
        action_name=action_name,
        tool_invocations=tool_invocations,
    )

    with default_span_factory.tool_call(
        tool_name=tool_name,
        tool_call_id=f"toolcall-{uuid.uuid4().hex[:12]}",
        tool_type=tool_type,
        tool_args=code_text,
        record_args=True,
    ) as ctx:
        try:
            ctx.span.update_name(f"llmmas.tool_call {tool_name}")
        except Exception:
            pass

        ctx.span.set_attribute("llmmas.tool.executor_kind", executor_kind)
        ctx.span.set_attribute("llmmas.tool.executor_class", type(executor).__name__)
        ctx.span.set_attribute("llmmas.tool.semantic_action", action_name)
        ctx.span.set_attribute("llmmas.tool.invocation.count", len(tool_invocations))
        ctx.span.set_attribute(
            "llmmas.tool.invocation.names",
            ",".join(sorted({item["tool_name"] for item in tool_invocations})),
        )
        ctx.span.set_attribute(
            "llmmas.tool.invocation.variables",
            ",".join(sorted({item["variable_name"] for item in tool_invocations})),
        )

        current_agent = _CURRENT_AGENT_NAME.get()
        if current_agent is not None:
            ctx.span.set_attribute("llmmas.agent.id", current_agent)

        if tool_invocations:
            first = tool_invocations[0]
            ctx.span.set_attribute("llmmas.tool.variable_name", first.get("variable_name", ""))
            if first.get("line") is not None:
                ctx.span.set_attribute("llmmas.tool.source.line", int(first["line"]))

        try:
            result = original(executor, code_blocks, *args, **kwargs)
            _annotate_tool_execution_result(ctx.span, result)
            return result
        except Exception as exc:
            ctx.span.record_exception(exc)
            ctx.span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def _tool_operation_name(
    *,
    executor_kind: str,
    action_name: str,
    tool_invocations: list[dict[str, Any]],
) -> str:
    if tool_invocations:
        return _short_tool_list([item["tool_name"] for item in tool_invocations])
    if action_name == "initialize_tool_environment":
        return "initialize_tool_environment"
    return action_name or executor_kind


def _tool_operation_type(
    *,
    executor_kind: str,
    action_name: str,
    tool_invocations: list[dict[str, Any]],
) -> str:
    if tool_invocations:
        return "hyperagent_tool"
    if action_name == "initialize_tool_environment":
        return "system_setup"
    if executor_kind == "docker_exec":
        return "docker_exec"
    if executor_kind == "jupyter_exec":
        return "python_exec"
    return executor_kind


def _annotate_tool_execution_result(span: Any, result: Any) -> None:
    exit_code = getattr(result, "exit_code", None)
    output = getattr(result, "output", None)
    code_file = getattr(result, "code_file", None)

    if exit_code is not None:
        span.set_attribute("llmmas.tool.exit_code", int(exit_code))
        if int(exit_code) != 0:
            span.set_status(Status(StatusCode.ERROR, f"exit_code={exit_code}"))

    if output is not None:
        output_text = _safe_str(output)
        span.set_attribute(semconv.ATTR_TOOL_RESULT_PREVIEW, output_text[:500])
        span.set_attribute(semconv.ATTR_TOOL_RESULT_SHA256, _sha256(output_text))

    if code_file is not None:
        span.set_attribute("llmmas.tool.code_file", _safe_str(code_file))


def _extract_tool_invocations_from_code(code_text: str) -> list[dict[str, Any]]:
    invocations = []
    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*_run\s*\(", code_text):
        variable_name = match.group(1)
        line = code_text[: match.start()].count("\n") + 1
        invocations.append(
            {
                "variable_name": variable_name,
                "tool_name": _tool_name_from_variable(variable_name),
                "line": line,
            }
        )

    seen = set()
    unique = []
    for item in invocations:
        key = (item.get("variable_name"), item.get("tool_name"), item.get("line"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _tool_name_from_variable(variable_name: str) -> str:
    mapping = {
        "code_search": "code_search",
        "go_to_def": "go_to_definition",
        "find_all_refs": "find_all_references",
        "get_all_symbols": "get_all_symbols",
        "get_folder_structure": "get_folder_structure",
        "open_file": "open_file",
        "find_file": "find_file",
        "editor": "editor_file",
        "open_file_gen": "open_file",
    }
    return mapping.get(variable_name, variable_name)


def _execution_action_name(
    *,
    executor_kind: str,
    code_text: str,
    tool_invocations: list[dict[str, Any]],
) -> str:
    if executor_kind == "jupyter_exec":
        if _looks_like_tool_initialization(code_text):
            return "initialize_tool_environment"
        if tool_invocations:
            return "tool_dispatch:" + _short_tool_list(
                [item["tool_name"] for item in tool_invocations]
            )
        return "python_code_execution"

    if executor_kind == "docker_exec":
        return _docker_action_name(code_text)

    return "code_execution"


def _looks_like_tool_initialization(code_text: str) -> bool:
    return (
        "from hyperagent.tools.tools import *" in code_text
        or "Initialize tools for navigation" in code_text
        or "Initialize tools for editing" in code_text
    )


def _docker_action_name(code_text: str) -> str:
    lowered = code_text.lower()
    if "pytest" in lowered:
        return "run_tests:pytest"
    if "runtests.py" in lowered:
        return "run_tests:project_runner"
    if "tox " in lowered or "\ntox" in lowered:
        return "run_tests:tox"
    if "python " in lowered or "python3 " in lowered:
        return "run_python_command"
    if "pip install" in lowered:
        return "install_dependency"
    return "bash_command"


def _short_tool_list(tool_names: list[str], *, max_items: int = 3) -> str:
    unique = []
    for name in tool_names:
        if name not in unique:
            unique.append(name)
    if len(unique) <= max_items:
        return "+".join(unique)
    return "+".join(unique[:max_items]) + f"+{len(unique) - max_items}_more"


# ---------------------------------------------------------------------------
# Generic helpers
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
    exc_info = (None, None, None)
    try:
        for ctx in contexts:
            entered = ctx.__enter__()
            stack.append((ctx, entered))
        yield
    except BaseException as exc:
        exc_info = (type(exc), exc, exc.__traceback__)
        raise
    finally:
        while stack:
            ctx, _ = stack.pop()
            ctx.__exit__(*exc_info)


@contextmanager
def _null_cm():
    yield


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


def _is_outer_manager(agent_name: str) -> bool:
    return agent_name in OUTER_MANAGER_NAMES


def _is_manager_name(agent_name: str) -> bool:
    return _is_outer_manager(agent_name) or agent_name in INNER_MANAGER_TO_PARENT


def _next_step_index(agent_name: str) -> int:
    session_id = _session_key()
    key = (session_id, agent_name)
    idx = _STEP_COUNTERS[key]
    _STEP_COUNTERS[key] += 1
    return idx


def _infer_source_agent_for_reply(
    target_agent_name: str,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Optional[str]:
    sender = kwargs.get("sender")
    if sender is not None:
        sender_name = _agent_name(sender)
        if not _is_manager_name(sender_name):
            return sender_name

    messages = _extract_messages(args, kwargs)

    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, Mapping):
            name = last.get("name")
            role = last.get("role")
            candidate = name or role

            if candidate and not _is_manager_name(str(candidate)):
                return str(candidate)

    if target_agent_name == "Planner":
        return _last_real_speaker_from_messages(messages) or "Admin"

    return None


def _extract_messages(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Optional[Any]:
    messages = kwargs.get("messages")
    if messages is None and args:
        if isinstance(args[0], list):
            messages = args[0]
    return messages


def _last_real_speaker_from_messages(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None

    for msg in reversed(messages):
        if not isinstance(msg, Mapping):
            continue

        candidate = msg.get("name") or msg.get("role")
        if not candidate:
            continue

        candidate = str(candidate)
        if candidate in {"user, admin", "user", "admin"}:
            return "Admin"
        if candidate in {"assistant", "system"}:
            continue
        if not _is_manager_name(candidate):
            return candidate

    return None


def _infer_last_message_content(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Optional[str]:
    messages = _extract_messages(args, kwargs)

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


def _infer_planner_delegate(content: str) -> Optional[str]:
    if not content:
        return None

    if "Navigator" in content or "Codebase Navigator" in content:
        return "Navigator"
    if "Editor" in content or "Codebase Editor" in content:
        return "Editor"
    if "Executor" in content or "Codebase Executor" in content:
        return "Executor"

    return None


def _infer_message_kind(content: Optional[str]) -> str:
    if not content:
        return "message"

    lowered = content.lower()
    if "final answer" in lowered or "terminate=true" in lowered:
        return "final_answer"
    if "subgoal" in lowered or "intern name" in lowered or "request" in lowered:
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
    direct = _first_nonempty(kwargs.get("model"), kwargs.get("request_model"), kwargs.get("engine"))
    if direct:
        return _safe_str(direct)

    for key in ("config", "llm_config", "params"):
        model = _find_model_in_object(kwargs.get(key))
        if model:
            return model

    for item in args:
        model = _find_model_in_object(item)
        if model:
            return model

    for attr in (
        "model",
        "_model",
        "config",
        "_config",
        "llm_config",
        "_llm_config",
        "config_list",
        "_config_list",
        "_clients",
        "clients",
    ):
        try:
            value = getattr(client, attr, None)
        except Exception:
            value = None
        model = _find_model_in_object(value)
        if model:
            return model

    env_model = os.environ.get("HYPERAGENT_MODEL")
    if env_model:
        return env_model

    return "unknown-model"


def _infer_llm_provider(args: tuple[Any, ...], kwargs: Mapping[str, Any], client: Any) -> str:
    direct = _first_nonempty(kwargs.get("api_type"), kwargs.get("provider"))
    if direct:
        return _safe_str(direct)

    for key in ("config", "llm_config", "params"):
        provider = _find_provider_in_object(kwargs.get(key))
        if provider:
            return provider

    for item in args:
        provider = _find_provider_in_object(item)
        if provider:
            return provider

    for attr in (
        "api_type",
        "provider",
        "config",
        "_config",
        "llm_config",
        "_llm_config",
        "config_list",
        "_config_list",
        "_clients",
        "clients",
    ):
        try:
            value = getattr(client, attr, None)
        except Exception:
            value = None
        provider = _find_provider_in_object(value)
        if provider:
            return provider

    env_provider = os.environ.get("HYPERAGENT_PROVIDER")
    if env_provider:
        return env_provider

    cls = type(client).__name__.lower()
    if "azure" in cls:
        return "azure_openai"
    if "openai" in cls:
        return "openai"
    return "unknown-provider"


def _find_model_in_object(value: Any, *, _depth: int = 0) -> Optional[str]:
    if value is None or _depth > 5:
        return None
    if isinstance(value, Mapping):
        for key in ("model", "request_model", "engine", "model_name", "model_name_or_path"):
            candidate = value.get(key)
            if candidate:
                return _safe_str(candidate)
        for key, item in value.items():
            if key == "messages":
                continue
            if isinstance(item, (Mapping, list, tuple)):
                model = _find_model_in_object(item, _depth=_depth + 1)
                if model:
                    return model
    if isinstance(value, (list, tuple)):
        for item in value:
            model = _find_model_in_object(item, _depth=_depth + 1)
            if model:
                return model
    for attr in ("model", "_model"):
        try:
            candidate = getattr(value, attr, None)
        except Exception:
            candidate = None
        if candidate:
            return _safe_str(candidate)
    return None


def _find_provider_in_object(value: Any, *, _depth: int = 0) -> Optional[str]:
    if value is None or _depth > 5:
        return None
    if isinstance(value, Mapping):
        for key in ("api_type", "provider"):
            candidate = value.get(key)
            if candidate:
                return _safe_str(candidate)
        base_url = _first_nonempty(value.get("base_url"), value.get("api_base"))
        if base_url:
            base_url_s = _safe_str(base_url).lower()
            if "openrouter" in base_url_s:
                return "openrouter"
            if "openai" in base_url_s:
                return "openai"
        for key, item in value.items():
            if key == "messages":
                continue
            if isinstance(item, (Mapping, list, tuple)):
                provider = _find_provider_in_object(item, _depth=_depth + 1)
                if provider:
                    return provider
    if isinstance(value, (list, tuple)):
        for item in value:
            provider = _find_provider_in_object(item, _depth=_depth + 1)
            if provider:
                return provider
    return None


def _first_nonempty(*values: Any) -> Optional[Any]:
    for value in values:
        if value not in (None, ""):
            return value
    return None


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
        for key in ("config", "llm_config", "params"):
            messages = _find_messages_in_object(kwargs.get(key))
            if messages is not None:
                break
    if messages is None:
        return None
    return _safe_str(messages)


def _find_messages_in_object(value: Any, *, _depth: int = 0) -> Optional[Any]:
    if value is None or _depth > 5:
        return None
    if isinstance(value, Mapping):
        if "messages" in value:
            return value["messages"]
        for item in value.values():
            if isinstance(item, (Mapping, list, tuple)):
                messages = _find_messages_in_object(item, _depth=_depth + 1)
                if messages is not None:
                    return messages
    if isinstance(value, list):
        if all(isinstance(item, Mapping) and "role" in item for item in value):
            return value
        for item in value:
            messages = _find_messages_in_object(item, _depth=_depth + 1)
            if messages is not None:
                return messages
    if isinstance(value, tuple):
        for item in value:
            messages = _find_messages_in_object(item, _depth=_depth + 1)
            if messages is not None:
                return messages
    return None


def _infer_llm_output(result: Any) -> Optional[str]:
    if result is None:
        return None

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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except Exception:
        return repr(value)