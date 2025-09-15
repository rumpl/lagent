# Agent Runtime Event Specification

## Overview

This specification defines a transport-agnostic event system for agent runtime
implementations. It describes the events that an agent runtime MUST emit during
its lifecycle, their structure, timing, and emission rules. This specification
is implementation-agnostic and can be implemented in any programming language or
transport mechanism.

## Event Structure

All events MUST implement the following base interface:

```
Event {
  type: string           // Event type identifier
  agent_name?: string    // Optional agent attribution
}
```

Some events include an `AgentContext` for attribution:

```
AgentContext {
  agent_name?: string    // Name of the agent associated with this event
}
```

## Event Types

### 1. Stream Lifecycle Events

#### StreamStartedEvent

**Purpose**: Signals the beginning of an agent runtime stream session.

**Structure**:

```json
{
  "type": "stream_started"
}
```

**Emission Rules**:

- MUST be the first event emitted (after optional user_message event)
- MUST be emitted once per stream session
- MUST occur before any agent processing begins

#### StreamStoppedEvent

**Purpose**: Signals the end of an agent runtime stream session.

**Structure**:

```json
{
  "type": "stream_stopped"
}
```

**Emission Rules**:

- MUST be the last event emitted in a stream session
- MUST be emitted once per stream session
- MUST occur after all processing is complete

### 2. User Interaction Events

#### UserMessageEvent

**Purpose**: Indicates when a user message is received by the runtime.

**Structure**:

```json
{
  "type": "user_message",
  "message": "string"
}
```

**Fields**:

- `message`: The complete user message content

**Emission Rules**:

- MUST be emitted when a user provides input to the system
- MUST be emitted before stream_started if the session includes user input
- SHOULD contain the exact user message without modification

### 3. Agent Response Events

#### AgentChoiceEvent

**Purpose**: Streams agent-generated content as it's produced.

**Structure**:

```json
{
  "type": "agent_choice",
  "content": "string",
  "agent_name": "string"
}
```

**Fields**:

- `content`: Incremental content chunk from the agent
- `agent_name`: Name of the generating agent

**Emission Rules**:

- MUST be emitted for each content delta/chunk from the agent
- MUST include agent attribution
- SHOULD preserve the order of content generation

#### AgentChoiceReasoningEvent

**Purpose**: Streams agent reasoning content (when available).

**Structure**:

```json
{
  "type": "agent_choice_reasoning",
  "content": "string",
  "agent_name": "string"
}
```

**Fields**:

- `content`: Incremental reasoning content chunk
- `agent_name`: Name of the reasoning agent

**Emission Rules**:

- MUST be emitted when agent produces reasoning content
- MUST include agent attribution
- MAY be interleaved with AgentChoiceEvent
- SHOULD preserve the order of reasoning generation

### 4. Tool Call Events

#### PartialToolCallEvent

**Purpose**: Indicates a tool call is being constructed (streaming).

**Structure**:

```json
{
  "type": "partial_tool_call",
  "tool_call": {
    "id": "string",
    "type": "function",
    "function": {
      "name": "string",
      "arguments": "string"
    }
  },
  "agent_name": "string"
}
```

**Fields**:

- `tool_call`: Partial tool call structure with available fields
- `agent_name`: Name of the calling agent

**Emission Rules**:

- MUST be emitted when a tool call starts being constructed
- MUST be emitted before complete tool call information is available
- SHOULD be emitted when function name becomes available
- MAY be emitted multiple times as arguments are streamed

#### ToolCallEvent

**Purpose**: Indicates a complete tool call is ready for execution.

**Structure**:

```json
{
  "type": "tool_call",
  "tool_call": {
    "id": "string",
    "type": "function",
    "function": {
      "name": "string",
      "arguments": "string"
    }
  },
  "agent_name": "string"
}
```

**Fields**:

- `tool_call`: Complete tool call structure
- `agent_name`: Name of the calling agent

**Emission Rules**:

- MUST be emitted when tool execution begins
- MUST include complete tool call information
- MUST be emitted after PartialToolCallEvent (if applicable)
- MUST precede ToolCallResponseEvent for the same tool call

#### ToolCallConfirmationEvent

**Purpose**: Requests user approval for tool execution.

**Structure**:

```json
{
  "type": "tool_call_confirmation",
  "tool_call": {
    "id": "string",
    "type": "function",
    "function": {
      "name": "string",
      "arguments": "string"
    }
  },
  "agent_name": "string"
}
```

**Fields**:

- `tool_call`: Tool call requiring confirmation
- `agent_name`: Name of the calling agent

**Emission Rules**:

- MUST be emitted when tool execution requires user approval
- MUST be emitted before tool execution begins
- MUST wait for user response before proceeding
- MAY be bypassed if auto-execution is enabled or tool is read-only

#### ToolCallResponseEvent

**Purpose**: Provides the result of tool execution.

**Structure**:

```json
{
  "type": "tool_call_response",
  "tool_call": {
    "id": "string",
    "type": "function",
    "function": {
      "name": "string",
      "arguments": "string"
    }
  },
  "response": "string",
  "agent_name": "string"
}
```

**Fields**:

- `tool_call`: The executed tool call
- `response`: Tool execution result or error message
- `agent_name`: Name of the calling agent

**Emission Rules**:

- MUST be emitted after tool execution completes
- MUST include the complete tool call that was executed
- MUST include the response (success or error)
- SHOULD include error messages for failed executions
- MUST maintain correspondence with ToolCallEvent via tool call ID

### 5. System Events

#### ErrorEvent

**Purpose**: Reports runtime errors and exceptions.

**Structure**:

```json
{
  "type": "error",
  "error": "string",
  "agent_name": "string"
}
```

**Fields**:

- `error`: Error message or description
- `agent_name`: Agent associated with the error (if applicable)

**Emission Rules**:

- MUST be emitted when runtime errors occur
- SHOULD include descriptive error messages
- MAY include stack traces or debug information
- SHOULD be emitted before stream termination on fatal errors

#### TokenUsageEvent

**Purpose**: Reports token consumption and cost tracking.

**Structure**:

```json
{
  "type": "token_usage",
  "usage": {
    "input_tokens": "integer",
    "output_tokens": "integer",
    "context_length": "integer",
    "context_limit": "integer",
    "cost": "number"
  },
  "agent_name": "string"
}
```

**Fields**:

- `usage.input_tokens`: Number of input tokens consumed
- `usage.output_tokens`: Number of output tokens generated
- `usage.context_length`: Current context length
- `usage.context_limit`: Maximum context limit
- `usage.cost`: Monetary cost of the operation
- `agent_name`: Agent responsible for token usage

**Emission Rules**:

- MUST be emitted after each agent interaction that consumes tokens
- SHOULD provide cumulative usage for the session
- MUST include cost information when available
- SHOULD be emitted after adding assistant messages to conversation

#### ShellOutputEvent

**Purpose**: Reports shell command execution output.

**Structure**:

```json
{
  "type": "shell",
  "output": "string"
}
```

**Fields**:

- `output`: Shell command output or error text

**Emission Rules**:

- MUST be emitted when shell commands are executed
- SHOULD include both stdout and stderr
- MAY be emitted multiple times for streaming output

### 6. Session Management Events

#### SessionTitleEvent

**Purpose**: Provides generated title for the conversation session.

**Structure**:

```json
{
  "type": "session_title",
  "session_id": "string",
  "title": "string",
  "agent_name": "string"
}
```

**Fields**:

- `session_id`: Unique session identifier
- `title`: Generated session title
- `agent_name`: Agent that generated the title

**Emission Rules**:

- SHOULD be emitted when a session title is generated
- MUST include unique session identifier
- SHOULD be emitted after conversation has sufficient content
- MAY be emitted at session end if no title exists

#### SessionSummaryEvent

**Purpose**: Provides generated summary of the conversation session.

**Structure**:

```json
{
  "type": "session_summary",
  "session_id": "string",
  "summary": "string",
  "agent_name": "string"
}
```

**Fields**:

- `session_id`: Unique session identifier
- `summary`: Generated session summary
- `agent_name`: Agent that generated the summary

**Emission Rules**:

- MUST be emitted when session summarization occurs
- SHOULD include comprehensive conversation summary
- MUST include unique session identifier
- SHOULD be emitted during session compaction

#### SessionCompactionEvent

**Purpose**: Indicates session compaction status (context management).

**Structure**:

```json
{
  "type": "session_compaction",
  "session_id": "string",
  "status": "string",
  "agent_name": "string"
}
```

**Fields**:

- `session_id`: Unique session identifier
- `status`: Compaction status ("start", "started", "completed")
- `agent_name`: Agent performing compaction

**Emission Rules**:

- MUST be emitted when session compaction begins ("start")
- MUST be emitted when compaction processing starts ("started")
- MUST be emitted when compaction completes ("completed")
- SHOULD occur when context approaches limits
- MUST include session identifier

## Event Emission Patterns

### Stream Session Lifecycle

1. `UserMessageEvent` (if user input present)
2. `StreamStartedEvent`
3. Multiple agent processing events
4. `StreamStoppedEvent`

### Agent Response Pattern

1. Multiple `AgentChoiceEvent` (content streaming)
2. Optional `AgentChoiceReasoningEvent` (reasoning streaming)
3. `TokenUsageEvent` (after response completion)

### Tool Execution Pattern

1. `PartialToolCallEvent` (optional, during streaming)
2. `ToolCallConfirmationEvent` (if approval required)
3. `ToolCallEvent` (execution start)
4. `ToolCallResponseEvent` (execution result)

### Session Compaction Pattern

1. `SessionCompactionEvent` (status: "start")
2. `SessionCompactionEvent` (status: "started")
3. `SessionSummaryEvent`
4. `TokenUsageEvent` (updated after compaction)
5. `SessionCompactionEvent` (status: "completed")

## Event Dependencies

### Required Sequences

- `StreamStartedEvent` MUST precede all processing events
- `StreamStoppedEvent` MUST follow all processing events
- `ToolCallEvent` MUST precede corresponding `ToolCallResponseEvent`
- `TokenUsageEvent` SHOULD follow agent content generation

### Optional Sequences

- `PartialToolCallEvent` MAY precede `ToolCallEvent`
- `ToolCallConfirmationEvent` MAY precede `ToolCallEvent`
- `SessionTitleEvent` MAY be emitted at session end
- `SessionSummaryEvent` SHOULD accompany compaction

### Agent Attribution

- Events with `agent_name` field MUST include valid agent identifier
- Agent attribution MUST be consistent within a tool call sequence
- Agent attribution SHOULD reflect the actual agent performing the action

## Implementation Requirements

### Mandatory Events

Implementations MUST support:

- `StreamStartedEvent`
- `StreamStoppedEvent`
- `AgentChoiceEvent`
- `ToolCallEvent`
- `ToolCallResponseEvent`
- `ErrorEvent`

### Optional Events

Implementations MAY support:

- `UserMessageEvent`
- `PartialToolCallEvent`
- `ToolCallConfirmationEvent`
- `AgentChoiceReasoningEvent`
- `TokenUsageEvent`
- `ShellOutputEvent`
- `SessionTitleEvent`
- `SessionSummaryEvent`
- `SessionCompactionEvent`

### Transport Requirements

- Events MUST be serializable to JSON
- Event order MUST be preserved in transport
- Event delivery MUST be reliable within a session
- Transport MUST support asynchronous event streaming

### Error Handling

- Invalid events SHOULD be logged and ignored
- Missing required fields SHOULD result in event rejection
- Transport errors SHOULD not terminate the session
- Fatal runtime errors MUST emit `ErrorEvent` before termination

## Extensibility

Implementations MAY define additional event types provided they:

- Follow the base event structure
- Use unique type identifiers
- Include appropriate agent attribution
- Maintain backward compatibility
- Document emission rules and timing

Custom events SHOULD follow the naming pattern: `vendor_event_name` or similar
namespace convention.
