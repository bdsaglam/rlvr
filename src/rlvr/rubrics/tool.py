from typing import Dict, List

from verifiers.parsers import XMLParser


def make_tool_use_reward_func(
    tool_tag: str = "tool",
    result_tag: str = "result",
):
    """
    Create a reward function that checks tool use success.

    Args:
        tool_tag: The XML tag used to identify tool calls in assistant messages
        result_tag: The XML tag used to identify results in environment responses
    """
    assistant_parser = XMLParser(fields=[tool_tag])
    env_parser = XMLParser(fields=[result_tag])

    def tool_use_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool use success.

        Uses XMLParser to identify proper tool calls.
        """

        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0

            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg["role"] == "assistant":
                    # Use parser to check for tool tag
                    parsed = assistant_parser.parse(msg["content"])
                    if getattr(parsed, tool_tag, None):
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "tool":
                            tool_attempts += 1
                            # Check response with env_parser
                            parsed_response = env_parser.parse(trajectory[i + 1]["content"])
                            if result := getattr(parsed_response, result_tag, None):
                                if not result.startswith("Error:"):
                                    successful_executions += 1

            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return 0.5 * (successful_executions / tool_attempts)

        return [check_execution(c) for c in completions]

    return tool_use_reward_func
