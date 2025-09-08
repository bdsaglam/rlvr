"""Helper utilities for enhanced W&B logging of full conversation trajectories."""

from typing import Any, Dict, Union

from verifiers.types import Messages


def sanitize_tool_calls(completion: list[dict[str, Any]] | str) -> list[dict[str, Any]] | str:
    if isinstance(completion, str):
        return completion
    for msg in completion:
        if "tool_calls" in msg:
            tool_calls = []
            msg["tool_calls"] = []
            for tc in msg["tool_calls"]:
                tool_calls.append(
                    {
                        "name": tc.get("function", {}).get("name", ""),
                        "args": tc.get("function", {}).get("arguments", {}),
                    }
                )
            msg["content"] += str({"tool_calls": tool_calls})
            msg.pop("tool_calls")
        if "tool_call_id" in msg:
            msg.pop("tool_call_id")
    return completion


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
        result = result[: length] + "\n\n... [TRUNCATED] ...\n\n" + result[-length:]

    return result


def extract_trajectory_stats(messages: Union[Messages, str]) -> Dict[str, Any]:
    """
    Extract statistics from a conversation trajectory.

    Args:
        messages: Conversation messages

    Returns:
        Dictionary with trajectory statistics
    """
    if isinstance(messages, str):
        return {"total_length": len(messages)}

    if not messages:
        return {"total_length": 0}

    stats = {
        "total_length": 0,
    }

    for msg in messages:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content", "")
        stats["total_length"] += len(str(content))

    return stats



def format_reward_components(rewards_dict: Dict[str, float]) -> str:
    """
    Format reward components for display.

    Args:
        rewards_dict: Dictionary of reward components

    Returns:
        Formatted string with reward breakdown
    """
    if not rewards_dict:
        return "No rewards"

    parts = []
    for key, value in rewards_dict.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.3f}")
        else:
            parts.append(f"{key}: {value}")

    return " | ".join(parts)
