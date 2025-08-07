from typing import List

from verifiers.rubrics.utils import get_last_answer


def int_answer_reward_func(completions, **kwargs) -> List[float]:
    """Reward function that checks if the final answer is an integer."""
    responses = [get_last_answer(c) for c in completions]
    return [1.0 if str(r).strip().isdigit() else 0.0 for r in responses]


def numeric_answer_reward_func(completions, **kwargs) -> List[float]:
    responses = [get_last_answer(c) for c in completions]
    return [1.0 if str(r).strip().isnumeric() else 0.0 for r in responses]


def safe_float(s: str, thousands_separator: str = ",") -> float | None:
    try:
        s = s.replace(thousands_separator, "")
        return float(s)
    except ValueError:
        return None


def numerical_equivalence_reward_func(completions, answer, **kwargs) -> List[float]:
    responses = [get_last_answer(c) for c in completions]
    return [1.0 if safe_float(str(r)) == safe_float(str(a)) else 0.0 for r, a in zip(responses, answer)]
