import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import (AIMessage, AIMessageChunk, HumanMessage,
                                     ToolMessage)
from langchain_core.runnables.config import RunnableConfig

from sample_agent import agent
from sample_agent.agent import AgentState, graph


# SSE Event Classes based on the specification
class BaseEvent:
    """Base event structure that all events must implement."""
    def __init__(self, type: str, agent_name: Optional[str] = None):
        self.type = type
        self.agent_name = agent_name

    def to_sse(self) -> str:
        """Convert event to SSE format."""
        data = {
            "type": self.type
        }
        if self.agent_name:
            data["agent_name"] = self.agent_name

        return f"data: {json.dumps(data)}\n\n"


class UserMessageEvent(BaseEvent):
    """Event emitted when a user message is received."""
    def __init__(self, message: str, agent_name: Optional[str] = None):
        super().__init__("user_message", agent_name)
        self.message = message

    def to_sse(self) -> str:
        data = {
            "type": self.type,
            "message": self.message
        }
        return f"data: {json.dumps(data)}\n\n"


class StreamStartedEvent(BaseEvent):
    """Event emitted at the beginning of a stream session."""
    def __init__(self, agent_name: Optional[str] = None):
        super().__init__("stream_started", agent_name)


class StreamStoppedEvent(BaseEvent):
    """Event emitted at the end of a stream session."""
    def __init__(self, agent_name: Optional[str] = None):
        super().__init__("stream_stopped", agent_name)


class AgentChoiceEvent(BaseEvent):
    """Event for streaming agent-generated content."""
    def __init__(self, content: str, agent_name: Optional[str] = None):
        super().__init__("agent_choice", agent_name)
        self.content = content

    def to_sse(self) -> str:
        data = {
            "type": self.type,
            "content": self.content
        }
        if self.agent_name:
            data["agent_name"] = self.agent_name
        return f"data: {json.dumps(data)}\n\n"


class AgentChoiceReasoningEvent(BaseEvent):
    """Event for streaming agent reasoning content."""
    def __init__(self, content: str, agent_name: Optional[str] = None):
        super().__init__("agent_choice_reasoning", agent_name)
        self.content = content

    def to_sse(self) -> str:
        data = {
            "type": self.type,
            "content": self.content
        }
        if self.agent_name:
            data["agent_name"] = self.agent_name
        return f"data: {json.dumps(data)}\n\n"


class ToolCallEvent(BaseEvent):
    """Event for complete tool calls ready for execution."""
    def __init__(self, tool_call: Dict[str, Any], agent_name: Optional[str] = None):
        super().__init__("tool_call", agent_name)
        self.tool_call = tool_call
    
    def to_sse(self) -> str:
        data = {
            "type": self.type,
            "tool_call": self.tool_call
        }
        if self.agent_name:
            data["agent_name"] = self.agent_name
        return f"data: {json.dumps(data)}\n\n"


class PartialToolCallEvent(BaseEvent):
    """Event for partial tool calls being constructed."""
    def __init__(self, tool_call: Dict[str, Any], agent_name: Optional[str] = None):
        super().__init__("partial_tool_call", agent_name)
        self.tool_call = tool_call

    def to_sse(self) -> str:
        data = {
            "type": self.type,
            "tool_call": self.tool_call
        }
        if self.agent_name:
            data["agent_name"] = self.agent_name
        return f"data: {json.dumps(data)}\n\n"


class ToolCallResponseEvent(BaseEvent):
    """Event for tool execution results."""
    def __init__(self, tool_call: Dict[str, Any], response: str, agent_name: Optional[str] = None):
        super().__init__("tool_call_response", agent_name)
        self.tool_call = tool_call
        self.response = response

    def to_sse(self) -> str:
        data = {
            "type": self.type,
            "tool_call": self.tool_call,
            "response": self.response
        }
        if self.agent_name:
            data["agent_name"] = self.agent_name
        return f"data: {json.dumps(data)}\n\n"


class ErrorEvent(BaseEvent):
    """Event for runtime errors and exceptions."""
    def __init__(self, error: str, agent_name: Optional[str] = None):
        super().__init__("error", agent_name)
        self.error = error

    def to_sse(self) -> str:
        data = {
            "type": self.type,
            "error": self.error
        }
        if self.agent_name:
            data["agent_name"] = self.agent_name
        return f"data: {json.dumps(data)}\n\n"


class TokenUsageEvent(BaseEvent):
    """Event for token consumption reporting."""
    def __init__(self, usage: Dict[str, Any], agent_name: Optional[str] = None):
        super().__init__("token_usage", agent_name)
        self.usage = usage

    def to_sse(self) -> str:
        data = {
            "type": self.type,
            "usage": self.usage
        }
        if self.agent_name:
            data["agent_name"] = self.agent_name
        return f"data: {json.dumps(data)}\n\n"


class ShellOutputEvent(BaseEvent):
    """Event for shell command execution output."""
    def __init__(self, output: str, agent_name: Optional[str] = None):
        super().__init__("shell", agent_name)
        self.output = output

    def to_sse(self) -> str:
        data = {
            "type": self.type,
            "output": self.output
        }
        return f"data: {json.dumps(data)}\n\n"


app = FastAPI()

class SSEEventEmitter:
    """Helper class for emitting SSE events during graph execution."""

    def __init__(self, agent_name: str = "sample_agent"):
        self.agent_name = agent_name
        self.events: List[BaseEvent] = []

    def emit_event(self, event: BaseEvent) -> None:
        """Add an event to the event list."""
        if not event.agent_name and hasattr(event, 'agent_name'):
            event.agent_name = self.agent_name
        self.events.append(event)

    def get_events(self) -> List[BaseEvent]:
        """Get all collected events."""
        return self.events.copy()


async def _handle_tool_message(
    tool_message: ToolMessage,
    agent_name: str
) -> AsyncGenerator[str, None]:
    """
    Handle ToolMessage responses and emit ToolCallResponseEvent.
    
    This function processes tool execution results and emits the appropriate
    ToolCallResponseEvent according to the specification.
    """
    try:
        # Extract tool call information from the ToolMessage
        tool_call_id = getattr(tool_message, 'tool_call_id', '')
        content = getattr(tool_message, 'content', '')
        
        # We need to reconstruct the tool call information
        # The tool call details might be in additional_kwargs or metadata
        tool_call_dict = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": getattr(tool_message, 'name', ''),
                "arguments": {}  # Arguments are from the original call, not in the response
            }
        }
        
        # If we have additional metadata about the original tool call, use it
        if hasattr(tool_message, 'additional_kwargs') and tool_message.additional_kwargs:
            if 'tool_call' in tool_message.additional_kwargs:
                tool_call_dict.update(tool_message.additional_kwargs['tool_call'])
        
        yield ToolCallResponseEvent(
            tool_call=tool_call_dict,
            response=content,
            agent_name=agent_name
        ).to_sse()
        
    except Exception as e:
        yield ErrorEvent(
            error=f"Error processing tool message: {str(e)}",
            agent_name=agent_name
        ).to_sse()


async def _handle_node_output(
    node_output: Union[AIMessageChunk, AIMessage], 
    agent_name: str
) -> AsyncGenerator[str, None]:
    """
    Handle outputs from the chat node and emit appropriate events based on the specification.
    
    This function analyzes the AIMessageChunk and emits the appropriate events:
    - AgentChoiceEvent for regular content
    - AgentChoiceReasoningEvent for reasoning content  
    - PartialToolCallEvent for partial tool calls
    - ToolCallEvent for complete tool calls
    - TokenUsageEvent for token usage information
    - ErrorEvent for any errors during processing
    """
    try:
        print(f"Processing node output: {node_output}")
        
        # Handle regular content
        if node_output.content and isinstance(node_output.content, str) and node_output.content.strip():
            yield AgentChoiceEvent(
                content=node_output.content,
                agent_name=agent_name
            ).to_sse()
        
        # Handle reasoning content (for models like o1 that support reasoning)
        if hasattr(node_output, 'reasoning') and node_output.reasoning:
            yield AgentChoiceReasoningEvent(
                content=node_output.reasoning,
                agent_name=agent_name
            ).to_sse()
        
        # Handle tool calls
        if hasattr(node_output, 'tool_calls') and node_output.tool_calls:
            for tool_call in node_output.tool_calls:
                # Convert tool call to the expected format
                tool_call_dict = {
                    "id": getattr(tool_call, 'id', ''),
                    "type": "function",
                    "function": {
                        "name": getattr(tool_call, 'name', ''),
                        "arguments": getattr(tool_call, 'args', {})
                    }
                }
                
                # Check if this is a complete tool call
                if (tool_call_dict["id"] and 
                    tool_call_dict["function"]["name"] and 
                    tool_call_dict["function"]["arguments"]):
                    # Complete tool call
                    yield ToolCallEvent(
                        tool_call=tool_call_dict,
                        agent_name=agent_name
                    ).to_sse()
                else:
                    # Partial tool call (streaming)
                    yield PartialToolCallEvent(
                        tool_call=tool_call_dict,
                        agent_name=agent_name
                    ).to_sse()
        
        # Handle tool call chunks (for streaming tool calls)
        if hasattr(node_output, 'tool_call_chunks') and node_output.tool_call_chunks:
            for chunk in node_output.tool_call_chunks:
                tool_call_dict = {
                    "id": getattr(chunk, 'id', ''),
                    "type": "function", 
                    "function": {
                        "name": getattr(chunk, 'name', ''),
                        "arguments": getattr(chunk, 'args', '')
                    }
                }
                
                yield PartialToolCallEvent(
                    tool_call=tool_call_dict,
                    agent_name=agent_name
                ).to_sse()
        
        # Handle token usage information
        if hasattr(node_output, 'usage_metadata') and node_output.usage_metadata:
            usage_data = {
                "input_tokens": getattr(node_output.usage_metadata, 'input_tokens', 0),
                "output_tokens": getattr(node_output.usage_metadata, 'output_tokens', 0),
                "total_tokens": getattr(node_output.usage_metadata, 'total_tokens', 0)
            }
            
            # Add cost if available
            if hasattr(node_output.usage_metadata, 'cost'):
                usage_data["cost"] = node_output.usage_metadata.cost
                
            yield TokenUsageEvent(
                usage=usage_data,
                agent_name=agent_name
            ).to_sse()
        
        # Handle response metadata (alternative token usage location)
        if hasattr(node_output, 'response_metadata') and node_output.response_metadata:
            if 'token_usage' in node_output.response_metadata:
                token_usage = node_output.response_metadata['token_usage']
                usage_data = {
                    "input_tokens": token_usage.get('prompt_tokens', 0),
                    "output_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0)
                }
                
                yield TokenUsageEvent(
                    usage=usage_data,
                    agent_name=agent_name
                ).to_sse()
                
    except Exception as e:
        # Emit error event for any processing errors
        yield ErrorEvent(
            error=f"Error processing node output: {str(e)}",
            agent_name=agent_name
        ).to_sse()


async def stream_graph_execution(
    messages: AgentState,
    thread_id: str = "default",
    agent_name: str = "sample_agent"
) -> AsyncGenerator[str, None]:
    """
    Stream SSE events during graph execution according to the specification.

    This function implements the event emission patterns defined in the spec:
    1. Stream session lifecycle
    2. Agent response patterns
    3. Tool execution patterns
    4. Error handling
    """

    try:
        # messages = graph_input.get("messages", [])
        # if messages:
        last_message = messages[-1]
        content = last_message["content"]

        if content:
            yield UserMessageEvent(message=content).to_sse()

        yield StreamStartedEvent(agent_name=agent_name).to_sse()

        config: RunnableConfig = {
            "configurable": {"thread_id": thread_id}
        }

        # Stream the graph execution
        async for chunk in graph.astream({"messages": messages}, config=config, stream_mode="messages"):
            try:
                print(chunk)
                (c, meta) = chunk
                
                # Handle different message types
                if isinstance(c, (AIMessageChunk, AIMessage)):
                    async for event_data in _handle_node_output(c, agent_name):
                        yield event_data
                elif isinstance(c, ToolMessage):
                    async for event_data in _handle_tool_message(c, agent_name):
                        yield event_data
                else:
                    # Handle other message types if needed
                    print(f"Unhandled message type: {type(c)}")

            except Exception as e:
                print(e)
                yield ErrorEvent(
                    error=f"Error processing chunk: {str(e)}",
                    agent_name=agent_name
                ).to_sse()

        yield StreamStoppedEvent(agent_name=agent_name).to_sse()

    except Exception as e:
        yield ErrorEvent(
            error=f"Fatal error during graph execution: {str(e)}",
            agent_name=agent_name
        ).to_sse()

        yield StreamStoppedEvent(agent_name=agent_name).to_sse()


async def run_handler(request: Request) -> StreamingResponse:
    """Run handler for the /cagent endpoint that streams SSE events."""
    try:
        body = await request.json()
    except Exception as e:
        async def error_stream():
            yield ErrorEvent(
                error=f"Failed to parse request body: {str(e)}",
                agent_name="sample_agent"
            ).to_sse()

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

    # print(body)

    # # Extract thread ID from request
    # thread_id = body.get("thread_id", "default")
    # agent_name = body.get("agent_name", "sample_agent")

    thread_id =" default"
    agent_name = "root"

    # Create the streaming response
    return StreamingResponse(
        stream_graph_execution(body, thread_id, agent_name),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

def agents_handler():
    """Handler for the /cagent/agents endpoint."""
    return [{"name": "sample_agent", "description": "An example agent to use as a starting point for your own agent.", "multi": False}]

sessions: Dict[str, Dict] = {}

def create_session_handler():
    """Handler for the /cagent/sessions endpoint."""
    session_id = f"session-{len(sessions) + 1}"
    sessions[session_id] = {"id": session_id, "messages": []}
    return sessions[session_id]


app.add_api_route("/api/agents", agents_handler, methods=["GET"])
app.add_api_route("/api/sessions", create_session_handler, methods=["POST"])
app.add_api_route("/api/sessions/{id}/agent/{name}", run_handler, methods=["POST"])

def main():
    """Run the uvicorn server."""
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "sample_agent.demo:app",
        host="0.0.0.0",
        port=port,
        reload=True,
    )

if __name__ == "__main__":
    main()