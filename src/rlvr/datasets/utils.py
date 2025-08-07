import random

from datasets import Dataset


def update_messages(
    messages: list[dict],
    system_prompt: str = None,
    few_shot: list[dict] = None,
    few_shot_prob: float = 1.0,
) -> list[dict]:
    # Order: system prompt, few-shot examples (user, assistant, tool, etc.), user message
    suffix = []
    if messages[-1]["role"] == "user":
        suffix.append(messages.pop())

    # Insert system prompt if it doesn't exist
    if system_prompt:
        if "system" in [m["role"] for m in messages]:
            raise ValueError("System prompt already exists")
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Add few-shot examples if they exist
    if few_shot and random.random() < few_shot_prob:
        messages.extend(few_shot)

    messages.extend(suffix)

    return messages


def prepare_example_for_env(
    example: dict,
    system_prompt: str = None,
    few_shot: list[dict] = None,
    few_shot_prob: float = 1.0,
) -> dict:
    example["prompt"] = update_messages(example["prompt"], system_prompt, few_shot, few_shot_prob)
    return example


def prepare_dataset_for_env(
    dataset: Dataset,
    system_prompt: str = None,
    few_shot: list[dict] = None,
    few_shot_prob: float = 1.0,
) -> Dataset:
    return dataset.map(lambda x: prepare_example_for_env(x, system_prompt, few_shot, few_shot_prob))
