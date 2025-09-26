"""Helper utilities for enhanced W&B logging of full conversation trajectories."""

import copy
from typing import Any, Union

from verifiers.types import ChatMessage, Messages


def sanitize_tool_calls(completion: list[list[ChatMessage]] | str) -> list[dict[str, Any]] | str:
    if isinstance(completion, str):
        return completion

    # Create a deep copy to avoid mutating the input
    sanitized_completion = copy.deepcopy(completion)

    for msg in sanitized_completion:
        if tool_calls := msg.get("tool_calls"):
            formatted_tool_calls = [
                {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in tool_calls
            ]
            msg["content"] += str({"tool_calls": formatted_tool_calls})
            msg.pop("tool_calls")
        msg.pop("tool_call_id", None)
    return sanitized_completion


def maybe_truncate_content(content: str, max_length: int | None = None) -> str:
    if max_length is None or len(content) <= max_length:
        return content
    else:
        length = max_length // 2
        return content[:length] + "\n... [TRUNCATED] ...\n" + content[-length:]


def format_conversation(messages: Union[Messages, str], max_length: int | None = None) -> str:
    """
    Format a full conversation trajectory for W&B logging.

    Args:
        messages: Either a string or list of message dictionaries
        max_length: Maximum length of formatted output

    Returns:
        Formatted string representation of the conversation
    """
    if isinstance(messages, str):
        return maybe_truncate_content(messages, max_length)

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

    content = "\n\n".join(formatted_parts)
    content = maybe_truncate_content(content, max_length)
    return content
