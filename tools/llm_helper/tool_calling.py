"""Structured Tool / Function Calling Engine.

Generates function calling declarations, tool calls, and execution dialogues:
- Formats OpenAPI 3.1 endpoints and CLI commands into tool schemas.
- Synthesizes user requests demanding action execution.
- Emits assistant turns containing standard `<tool_call>...</tool_call>` JSON payloads.
- Supports mock tool execution responses `<tool_response>...</tool_response>`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF\uFE0F]",
    flags=re.UNICODE,
)


@dataclass(frozen=True)
class ToolExecutionPayload:
    """Invocation specification for structured tool call record creation."""

    tool_def: dict[str, Any]
    function_name: str
    call_arguments: dict[str, Any]
    mock_result: dict[str, Any]
    final_assistant_summary: str
    category: str = "api_procedure"


class ToolCallingFormatter:
    """Formats function definitions and tool call dialogues."""

    def __init__(self, format_style: str = "hermes") -> None:
        self.format_style = format_style

    def format_tool_declaration(self, tool_def: dict[str, Any]) -> str:
        """Render a tool declaration in the configured schema."""
        tool_json = json.dumps(tool_def, indent=2)
        return f"<tools>\n{tool_json}\n</tools>"

    def format_tool_call(self, function_name: str, arguments: dict[str, Any]) -> str:
        """Format an invocation payload."""
        payload = {"name": function_name, "arguments": arguments}
        return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"

    def format_tool_response(self, function_name: str, response: dict[str, Any]) -> str:
        """Format a tool return result."""
        payload = {"name": function_name, "content": response}
        return f"<tool_response>\n{json.dumps(payload)}\n</tool_response>"

    def build_tool_execution_record(
        self,
        record_id: str,
        system_prompt: str,
        user_query: str,
        payload: ToolExecutionPayload,
    ) -> dict[str, Any]:
        """Synthesize a complete tool execution conversation."""
        tool_block = self.format_tool_declaration(payload.tool_def)
        full_system = f"{system_prompt}\n\n{tool_block}"

        t2_call = self.format_tool_call(payload.function_name, payload.call_arguments)
        t3_resp = self.format_tool_response(payload.function_name, payload.mock_result)

        action_turns = [
            ("user", user_query),
            ("assistant", t2_call),
            ("user", t3_resp),
            ("assistant", payload.final_assistant_summary),
        ]
        messages = [{"role": "system", "content": _EMOJI_PATTERN.sub("", full_system).strip()}]
        for role, body in action_turns:
            messages.append({"role": role, "content": _EMOJI_PATTERN.sub("", body).strip()})

        return {
            "id": record_id,
            "category": payload.category,
            "messages": messages,
            "metadata": {
                "has_tool_call": True,
                "tool_name": payload.function_name,
                "format_style": self.format_style,
            },
        }
