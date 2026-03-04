# Span names
SPAN_SESSION = "llmmas.session"
SPAN_SEGMENT = "llmmas.workflow.segment"
SPAN_AGENT_STEP = "agent_step"

# A2A messaging span naming:
A2A_OP_SEND = "send"
A2A_OP_PROCESS = "process"

# Core attributes (LLM-MAS specific)
ATTR_SESSION_ID = "llmmas.session.id"

ATTR_SEGMENT_NAME = "llmmas.segment.name"
ATTR_SEGMENT_ORDER = "llmmas.segment.order"
ATTR_SEGMENT_ORIGIN = "llmmas.segment.origin"

ATTR_AGENT_ID = "llmmas.agent.id"
ATTR_STEP_INDEX = "llmmas.step.index"

ATTR_SOURCE_AGENT_ID = "llmmas.source_agent.id"
ATTR_TARGET_AGENT_ID = "llmmas.target_agent.id"
ATTR_EDGE_ID = "llmmas.edge.id"
ATTR_MESSAGE_ID = "llmmas.message.id"
ATTR_CHANNEL = "llmmas.channel"

# Content-related (safe defaults)
ATTR_MESSAGE_PREVIEW = "llmmas.message.preview"
ATTR_MESSAGE_SHA256 = "llmmas.message.sha256"

# Fault injection
ATTR_FAULT_INJECTED = "llmmas.fault.injected"
ATTR_FAULT_TYPE = "llmmas.fault.type"
ATTR_FAULT_SPEC_ID = "llmmas.fault.spec_id"
ATTR_FAULT_DECISION = "llmmas.fault.decision"

# GenAI semantic conventions (core)
ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
ATTR_GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
ATTR_GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
ATTR_GEN_AI_REQUEST_ID = "gen_ai.request.id"

GEN_AI_OPERATION_EXECUTE_TOOL = "execute_tool"
GEN_AI_OPERATION_INFERENCE = "inference"

ATTR_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
ATTR_GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
ATTR_GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# Optional lightweight tool payload hints
ATTR_TOOL_ARGS_PREVIEW = "llmmas.tool.args.preview"
ATTR_TOOL_ARGS_SHA256 = "llmmas.tool.args.sha256"
ATTR_TOOL_RESULT_PREVIEW = "llmmas.tool.result.preview"
ATTR_TOOL_RESULT_SHA256 = "llmmas.tool.result.sha256"

# Optional lightweight LLM payload hints
ATTR_LLM_INPUT_PREVIEW = "llmmas.llm.input.preview"
ATTR_LLM_INPUT_SHA256 = "llmmas.llm.input.sha256"
ATTR_LLM_OUTPUT_PREVIEW = "llmmas.llm.output.preview"
ATTR_LLM_OUTPUT_SHA256 = "llmmas.llm.output.sha256"