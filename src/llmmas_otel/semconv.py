from __future__ import annotations

# Span names
SPAN_SESSION = "llmmas.session"
SPAN_WORKFLOW = "llmmas.workflow"
SPAN_SEGMENT = "llmmas.workflow.segment"  # backward-compatible alias-style span name
SPAN_AGENT_STEP = "llmmas.agent_step"
SPAN_DELEGATION = "llmmas.delegation"
SPAN_ENVIRONMENT_ACTION = "llmmas.environment_action"
SPAN_ARTIFACT = "llmmas.artifact"

# A2A messaging span naming
A2A_OP_SEND = "send"
A2A_OP_PROCESS = "process"

# Core attributes
ATTR_SESSION_ID = "llmmas.session.id"
ATTR_SESSION_NAME = "llmmas.session.name"
ATTR_SESSION_TASK_ID = "llmmas.session.task.id"
ATTR_FRAMEWORK = "llmmas.framework"
ATTR_SYSTEM = "llmmas.system"
ATTR_ADAPTER = "llmmas.adapter"

# Workflow / phase / segment attributes
ATTR_WORKFLOW_ID = "llmmas.workflow.id"
ATTR_WORKFLOW_NAME = "llmmas.workflow.name"
ATTR_WORKFLOW_KIND = "llmmas.workflow.kind"
ATTR_WORKFLOW_ORDER = "llmmas.workflow.order"
ATTR_WORKFLOW_ORIGIN = "llmmas.workflow.origin"
ATTR_WORKFLOW_PARENT_ID = "llmmas.workflow.parent_id"
ATTR_WORKFLOW_DEPTH = "llmmas.workflow.depth"

# Backward-compatible segment attributes
ATTR_SEGMENT_NAME = "llmmas.segment.name"
ATTR_SEGMENT_ORDER = "llmmas.segment.order"
ATTR_SEGMENT_ORIGIN = "llmmas.segment.origin"

# Agent step attributes
ATTR_AGENT_ID = "llmmas.agent.id"
ATTR_AGENT_ROLE = "llmmas.agent.role"
ATTR_AGENT_IMPL = "llmmas.agent.impl"
ATTR_PARENT_AGENT_ID = "llmmas.agent.parent_id"
ATTR_STEP_INDEX = "llmmas.step.index"
ATTR_STEP_KIND = "llmmas.step.kind"

# Routed message / A2A attributes
ATTR_SOURCE_AGENT_ID = "llmmas.source_agent.id"
ATTR_TARGET_AGENT_ID = "llmmas.target_agent.id"
ATTR_EDGE_ID = "llmmas.edge.id"
ATTR_MESSAGE_ID = "llmmas.message.id"
ATTR_MESSAGE_PARENT_ID = "llmmas.message.parent_id"
ATTR_MESSAGE_KIND = "llmmas.message.kind"
ATTR_MESSAGE_ROUTE_VIA = "llmmas.message.route.via"
ATTR_MESSAGE_DIRECTION = "llmmas.message.direction"
ATTR_CHANNEL = "llmmas.channel"

# Content-related safe defaults
ATTR_MESSAGE_PREVIEW = "llmmas.message.preview"
ATTR_MESSAGE_SHA256 = "llmmas.message.sha256"

# Delegation / subtask attributes
ATTR_DELEGATION_ID = "llmmas.delegation.id"
ATTR_DELEGATION_FROM_AGENT = "llmmas.delegation.from_agent"
ATTR_DELEGATION_TO_AGENT = "llmmas.delegation.to_agent"
ATTR_DELEGATION_TASK_ID = "llmmas.delegation.task.id"
ATTR_DELEGATION_KIND = "llmmas.delegation.kind"
ATTR_DELEGATION_VIA = "llmmas.delegation.via"
ATTR_DELEGATION_GOAL_PREVIEW = "llmmas.delegation.goal.preview"
ATTR_DELEGATION_GOAL_SHA256 = "llmmas.delegation.goal.sha256"

# Environment action attributes. A tool call is a subtype of environment action.
ATTR_ENV_ACTION_ID = "llmmas.env_action.id"
ATTR_ENV_ACTION_KIND = "llmmas.env_action.kind"
ATTR_ENV_ACTION_NAME = "llmmas.env_action.name"
ATTR_ENV_ACTION_INPUT_PREVIEW = "llmmas.env_action.input.preview"
ATTR_ENV_ACTION_INPUT_SHA256 = "llmmas.env_action.input.sha256"
ATTR_ENV_ACTION_OUTPUT_PREVIEW = "llmmas.env_action.output.preview"
ATTR_ENV_ACTION_OUTPUT_SHA256 = "llmmas.env_action.output.sha256"
ATTR_ENV_ACTION_EXIT_CODE = "llmmas.env_action.exit_code"
ATTR_ENV_ACTION_CHANGED_FILES = "llmmas.env_action.changed_files"

# Artifact / state-delta attributes
ATTR_ARTIFACT_ID = "llmmas.artifact.id"
ATTR_ARTIFACT_KIND = "llmmas.artifact.kind"
ATTR_ARTIFACT_NAME = "llmmas.artifact.name"
ATTR_ARTIFACT_PATH = "llmmas.artifact.path"
ATTR_ARTIFACT_SHA256 = "llmmas.artifact.sha256"
ATTR_ARTIFACT_SIZE_BYTES = "llmmas.artifact.size.bytes"

# Fault injection
ATTR_FAULT_INJECTED = "llmmas.fault.injected"
ATTR_FAULT_TYPE = "llmmas.fault.type"
ATTR_FAULT_SPEC_ID = "llmmas.fault.spec_id"
ATTR_FAULT_DECISION = "llmmas.fault.decision"

# GenAI semantic conventions
ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
ATTR_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_REQUEST_ID = "gen_ai.request.id"

GEN_AI_OPERATION_EXECUTE_TOOL = "execute_tool"
GEN_AI_OPERATION_INFERENCE = "inference"

ATTR_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
ATTR_GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
ATTR_GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# Optional lightweight tool payload hints. Kept for backward compatibility.
ATTR_TOOL_ARGS_PREVIEW = "llmmas.tool.args.preview"
ATTR_TOOL_ARGS_SHA256 = "llmmas.tool.args.sha256"
ATTR_TOOL_RESULT_PREVIEW = "llmmas.tool.result.preview"
ATTR_TOOL_RESULT_SHA256 = "llmmas.tool.result.sha256"

# Optional lightweight LLM payload hints
ATTR_LLM_INPUT_PREVIEW = "llmmas.llm.input.preview"
ATTR_LLM_INPUT_SHA256 = "llmmas.llm.input.sha256"
ATTR_LLM_OUTPUT_PREVIEW = "llmmas.llm.output.preview"
ATTR_LLM_OUTPUT_SHA256 = "llmmas.llm.output.sha256"