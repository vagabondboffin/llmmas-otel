# Span names
SPAN_SESSION = "llmmas.session"
SPAN_SEGMENT = "llmmas.workflow.segment"
SPAN_AGENT_STEP = "agent_step"

# A2A messaging span naming follows the common pattern:
#   "{operation} {destination}"
# where operation is typically "send" (producer) or "process" (consumer).
A2A_OP_SEND = "send"
A2A_OP_PROCESS = "process"

# Core attributes (LLM-MAS specific)
ATTR_SESSION_ID = "llmmas.session.id"

ATTR_SEGMENT_NAME = "llmmas.segment.name"
ATTR_SEGMENT_ORDER = "llmmas.segment.order"  # proposal term (ordering within session)
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

# Tool / environment interaction (GenAI semantic conventions where possible)
ATTR_GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_OPERATION_EXECUTE_TOOL = "execute_tool"

ATTR_GEN_AI_TOOL_NAME = "gen_ai.tool.name"
ATTR_GEN_AI_TOOL_TYPE = "gen_ai.tool.type"
ATTR_GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# Optional (llmmas.*) tool payload hints (kept lightweight by default)
ATTR_TOOL_ARGS_PREVIEW = "llmmas.tool.args.preview"
ATTR_TOOL_ARGS_SHA256 = "llmmas.tool.args.sha256"
ATTR_TOOL_RESULT_PREVIEW = "llmmas.tool.result.preview"
ATTR_TOOL_RESULT_SHA256 = "llmmas.tool.result.sha256"
