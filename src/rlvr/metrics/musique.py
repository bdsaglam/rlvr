import collections
import re
import string
from typing import Callable


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list[str]:
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact_match(a_gold: str, a_pred: str) -> int:
    """Compute the Exact Match (EM) score between a gold answer and a prediction."""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold: str, a_pred: str) -> float:
    """Compute the F1 score between a gold answer and a prediction."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    ground_truths: list[str],
) -> float:
    """Calculate the maximum metric score for a prediction over all ground truths."""
    scores_for_ground_truths = [metric_fn(prediction, ground_truth) for ground_truth in ground_truths]
    return max(scores_for_ground_truths)


def exact_match(prediction: str | None, reference: list[str]) -> float:
    if prediction is None:
        return 0.0
    return metric_max_over_ground_truths(compute_exact_match, prediction, reference)


def f1(prediction: str | None, reference: list[str]) -> float:
    if prediction is None:
        return 0.0
    return metric_max_over_ground_truths(compute_f1, prediction, reference)


def compute_metrics(prediction: str, reference: list[str]) -> dict[str, float]:
    return {
        "exact_match": exact_match(prediction, reference),
        "f1": f1(prediction, reference),
    }
