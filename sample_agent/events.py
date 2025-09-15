import json
from typing import Any, Dict, Optional


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
        # Ensure arguments are sent as strings
        tool_call_copy = self.tool_call.copy()
        if "function" in tool_call_copy and "arguments" in tool_call_copy["function"]:
            args = tool_call_copy["function"]["arguments"]
            # Convert to string based on type
            if isinstance(args, str):
                # Already a string, keep as-is
                pass
            elif isinstance(args, dict) and not args:
                # Empty dict becomes empty string
                tool_call_copy["function"]["arguments"] = ""
            else:
                # Non-empty dict or other types get JSON-encoded
                tool_call_copy["function"]["arguments"] = json.dumps(args)

        data = {
            "type": self.type,
            "tool_call": tool_call_copy
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
        # Ensure arguments are sent as strings
        tool_call_copy = self.tool_call.copy()
        if "function" in tool_call_copy and "arguments" in tool_call_copy["function"]:
            args = tool_call_copy["function"]["arguments"]
            # Convert to string based on type
            if isinstance(args, str):
                # Already a string, keep as-is
                pass
            elif isinstance(args, dict) and not args:
                # Empty dict becomes empty string
                tool_call_copy["function"]["arguments"] = ""
            else:
                # Non-empty dict or other types get JSON-encoded
                tool_call_copy["function"]["arguments"] = json.dumps(args)

        data = {
            "type": self.type,
            "tool_call": tool_call_copy
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
        # Ensure arguments are sent as strings
        tool_call_copy = self.tool_call.copy()
        if "function" in tool_call_copy and "arguments" in tool_call_copy["function"]:
            args = tool_call_copy["function"]["arguments"]
            # Convert to string based on type
            if isinstance(args, str):
                # Already a string, keep as-is
                pass
            elif isinstance(args, dict) and not args:
                # Empty dict becomes empty string
                tool_call_copy["function"]["arguments"] = ""
            else:
                # Non-empty dict or other types get JSON-encoded
                tool_call_copy["function"]["arguments"] = json.dumps(args)

        data = {
            "type": self.type,
            "tool_call": tool_call_copy,
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

