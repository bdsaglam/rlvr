from typing import Dict, List

from verifiers.models import RewardFunc
from verifiers.parsers import XMLParser


def make_code_execution_reward_func(code_tag: str = "code", output_tag: str = "output") -> RewardFunc:

    assistant_parser = XMLParser(fields=[code_tag])
    env_parser = XMLParser(fields=[output_tag])


    def code_execution_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """Reward function that checks code execution success at each step."""

        def check_execution(trajectory: List[Dict[str, str]]) -> float:
            total_code_steps = 0
            successful_executions = 0

            for i, msg in enumerate(trajectory):
                if msg["role"] == "assistant":
                    parsed = assistant_parser.parse(msg["content"])
                    if getattr(parsed, code_tag, None):
                        total_code_steps += 1
                        # Look for the next user message (environment response)
                        if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "user":
                            env_response = trajectory[i + 1]["content"]
                            parsed_response = env_parser.parse(env_response)
                            if output := getattr(parsed_response, output_tag, None):
                                if len(output) > 0 and not output.startswith("Error:"):
                                    successful_executions += 1

            # Return proportional reward based on successful executions
            if total_code_steps == 0:
                return 0.0
            return 0.3 * (successful_executions / total_code_steps) + 0.05 * (successful_executions)

        return [check_execution(c) for c in completions]

    return code_execution_reward_func