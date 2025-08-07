from typing import Callable, Dict, List

import numpy as np

from verifiers.parsers.xml_parser import XMLParser


def make_reasoning_reward_func(
    think_tag: str = "think",
    min_char_length: int = 300,
    max_char_length: int = 10000,
) -> Callable:
    parser = XMLParser(fields=[think_tag])

    def reasoning_reward_func(
        completions: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[float]:
        rewards = []
        for trajectory in completions:
            trajectory_rewards = []
            for message in trajectory:
                if message["role"] != "assistant":
                    continue
                parsed_response = parser.parse(message["content"])
                if result := getattr(parsed_response, think_tag, None):
                    # Reward based on the length of the think tag
                    # Penalize too short or too long think tags
                    if len(result) < min_char_length:
                        trajectory_rewards.append(0.1)
                    elif len(result) > max_char_length:
                        trajectory_rewards.append(0.1)
                    else:
                        trajectory_rewards.append(0.4)
                else:
                    trajectory_rewards.append(0.0)

            rewards.append(float(np.mean(trajectory_rewards)))

        return rewards

    return reasoning_reward_func
