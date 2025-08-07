from typing import List

from verifiers.rubrics.utils import get_last_answer


def exact_answer_reward_func(completions, answer, **kwargs) -> List[float]:
    """Reward function that checks if the final answer matches the expected answer."""
    responses = [get_last_answer(c) or "" for c in completions]
    return [1.0 if str(r).strip() == str(a).strip() else 0.0 for r, a in zip(responses, answer)]
