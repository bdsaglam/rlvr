"""Helper utilities for enhanced W&B logging of full conversation trajectories."""

import copy
from typing import Any, Dict, Union

from verifiers.types import Messages


def sanitize_tool_calls(completion: list[dict[str, Any]] | str) -> list[dict[str, Any]] | str:
    if isinstance(completion, str):
        return completion

    # Create a deep copy to avoid mutating the input
    sanitized_completion = copy.deepcopy(completion)

    for msg in sanitized_completion:
        if tool_calls := msg.get("tool_calls"):
            formatted_tool_calls = [
                {
                    "name": tc.get("function", {}).get("name", ""),
                    "args": tc.get("function", {}).get("arguments", {}),
                }
                for tc in tool_calls
            ]
            msg["content"] += str({"tool_calls": formatted_tool_calls})
            msg.pop("tool_calls")
        msg.pop("tool_call_id", None)
    return sanitized_completion


def format_conversation(messages: Union[Messages, str], max_length: int = 10000) -> str:
    """
    Format a full conversation trajectory for W&B logging.

    Args:
        messages: Either a string or list of message dictionaries
        max_length: Maximum length of formatted output

    Returns:
        Formatted string representation of the conversation
    """
    if isinstance(messages, str):
        return messages[:max_length]

    if not messages:
        return ""

    messages = sanitize_tool_calls(messages)

    formatted_parts = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Format the role header
        if role == "system":
            formatted_parts.append(f"⚙️ SYSTEM:\n{content}")
        elif role == "user":
            formatted_parts.append(f"👤 USER:\n{content}")
        elif role == "assistant":
            formatted_parts.append(f"🤖 ASSISTANT:\n{content}")
        elif role == "tool":
            formatted_parts.append(f"🛠️ TOOL:\n{content}")
        else:
            formatted_parts.append(f"[{role.upper()}]:\n{content}")

    result = "\n\n".join(formatted_parts)

    # Truncate if too long
    if len(result) > max_length:
        length = max_length // 2
        result = result[:length] + "\n\n... [TRUNCATED] ...\n\n" + result[-length:]

    return result
