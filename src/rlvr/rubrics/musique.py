from typing import List

from verifiers.metrics.musique import exact_match, f1
from verifiers.models import Message
from verifiers.rubrics.utils import get_last_answer
from verifiers.tools.retrieve import extract_all_retrieved_doc_ids


def musique_em_reward_func(
    completions: List[List[Message]],
    answers: list[list[str]],
    n_hops: list[int],
    **kwargs,
) -> List[float]:
    """
    Reward function that checks question answering success.
    """
    predicted_answers = [get_last_answer(c) or "" for c in completions]
    return [
        exact_match(predicted_answer, references) * _n_hops / 2
        for predicted_answer, references, _n_hops in zip(predicted_answers, answers, n_hops)
    ]


def musique_f1_reward_func(
    completions: List[List[Message]],
    answers: list[list[str]],
    n_hops: list[int],
    **kwargs,
) -> List[float]:
    predicted_answers = [get_last_answer(c) or "" for c in completions]
    return [
        f1(predicted_answer, references) * _n_hops / 2
        for predicted_answer, references, _n_hops in zip(predicted_answers, answers, n_hops)
    ]


def musique_supporting_recall_reward_func(
    completions: List[List[Message]],
    docs: list[list[dict]],
    **kwargs,
) -> List[float]:
    rewards = []
    for completion, _docs in zip(completions, docs):
        supporting_doc_ids = [doc["id"] for doc in _docs if doc["is_supporting"]]
        retrieved_doc_ids = set(extract_all_retrieved_doc_ids(completion))
        if len(retrieved_doc_ids) == 0:
            rewards.append(0.0)
            continue
        recall = len(retrieved_doc_ids & set(supporting_doc_ids)) / len(supporting_doc_ids)
        rewards.append(recall)
    return rewards


def musique_supporting_f1_reward_func(
    completions: List[List[Message]],
    docs: list[list[dict]],
    **kwargs,
) -> List[float]:
    rewards = []
    for completion, _docs in zip(completions, docs):
        supporting_doc_ids = [doc["id"] for doc in _docs if doc["is_supporting"]]
        retrieved_doc_ids = set(extract_all_retrieved_doc_ids(completion))
        if len(retrieved_doc_ids) == 0:
            rewards.append(0.0)
            continue
        precision = len(retrieved_doc_ids & set(supporting_doc_ids)) / len(retrieved_doc_ids)
        recall = len(retrieved_doc_ids & set(supporting_doc_ids)) / len(supporting_doc_ids)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        rewards.append(f1)
    return rewards
