import json
import re

from .metrics import exact_match, f1


def parse_structured_output(completion) -> dict:
    """Get the structured output of the last tool call."""
    if isinstance(completion, list) and len(completion) > 0:
        tool_calls = completion[-1].get("tool_calls", [])
        if len(tool_calls) > 0:
            try:
                return json.loads(tool_calls[-1].function.arguments)
            except (json.JSONDecodeError, AttributeError, KeyError):
                return {}
    return {}


def extract_retrieved_doc_ids(content: str) -> list[str]:
    """Extract document IDs from tool response content."""
    return [id.strip() for id in re.findall(r"^Document ID: (\S+)", content, re.MULTILINE)]


def extract_all_retrieved_doc_ids(completion):
    """Extract all retrieved document IDs from a completion."""
    retrieved_ids = set()

    if isinstance(completion, list):
        for message in completion:
            if message.get("role") == "tool":
                content = message.get("content", "")
                ids = extract_retrieved_doc_ids(content)
                retrieved_ids.update(ids)
    elif isinstance(completion, str):
        ids = extract_retrieved_doc_ids(completion)
        retrieved_ids.update(ids)

    return list(retrieved_ids)


# Reward functions for MuSiQue evaluation
def exact_match_reward(completion, answer, info, parser, **kwargs):
    """Exact match reward function."""
    predicted_answer = parse_structured_output(completion).get("final_answer", "")
    return exact_match(predicted_answer, info["answers"])


def f1_reward(completion, answer, info, parser, **kwargs):
    """F1 score reward function."""
    predicted_answer = parse_structured_output(completion).get("final_answer", "")
    return f1(predicted_answer, info["answers"])


def retrieval_recall_reward(completion, info, **kwargs):
    """Retrieval recall reward function."""
    supporting_doc_ids = [doc["id"] for doc in info["docs"] if doc["is_supporting"]]
    retrieved_doc_ids = set(extract_all_retrieved_doc_ids(completion))

    if len(retrieved_doc_ids) > 0 and len(supporting_doc_ids) > 0:
        return len(retrieved_doc_ids & set(supporting_doc_ids)) / len(supporting_doc_ids)
    return 0.0


def retrieval_precision_reward(completion, info, **kwargs):
    """Retrieval precision reward function."""
    supporting_doc_ids = [doc["id"] for doc in info["docs"] if doc["is_supporting"]]
    retrieved_doc_ids = set(extract_all_retrieved_doc_ids(completion))

    if len(retrieved_doc_ids) > 0 and len(supporting_doc_ids) > 0:
        return len(retrieved_doc_ids & set(supporting_doc_ids)) / len(retrieved_doc_ids)
    return 0.0


def extract_citations(completion, parser, cite_tag="cite"):
    """Extract citations from the completion using structured output parser."""
    return parse_structured_output(completion).get("cited_doc_ids", [])


def citation_reward(completion, info, parser, **kwargs):
    """Citation reward function based on verifiers citation.py pattern."""
    cited_doc_ids = extract_citations(completion, parser, cite_tag="cite")
    supporting_doc_ids = [doc["id"] for doc in info["docs"] if doc["is_supporting"]]

    if len(supporting_doc_ids) == 0:
        return 1.0  # Perfect score if no citations needed

    if len(cited_doc_ids) == 0:
        return 0.0  # No citations provided

    # Calculate citation recall: how many supporting docs were cited
    cited_supporting = len(set(cited_doc_ids) & set(supporting_doc_ids))
    return cited_supporting / len(supporting_doc_ids)


def format_reward(completion, parser, **kwargs):
    """Format reward function that rewards proper use of cite, think, and answer tags."""
    output = parse_structured_output(completion)
    fields = ["reasoning", "cited_doc_ids", "final_answer"]
    score = 0.0
    for field in fields:
        if output.get(field, None) is not None:
            score += 1
    return score / len(fields)


def combined_reward(*args, **kwargs):
    """Combined reward function."""
    # Get weighted scores, retrieval recall, and citation reward
    em_reward = exact_match_reward(*args, **kwargs)
    _f1_reward = f1_reward(*args, **kwargs)
    retrieval_recall = retrieval_recall_reward(*args, **kwargs)
    retrieval_precision = retrieval_precision_reward(*args, **kwargs)
    citation_score = citation_reward(*args, **kwargs)
    format_score = format_reward(*args, **kwargs)

    # Combine: average of weighted EM and F1, plus retrieval and citation bonuses
    pairs = [
        (em_reward, 0.9),
        (_f1_reward, 1.0),
        (retrieval_recall, 1.0),
        (retrieval_precision, 0.4),
        (citation_score, 0.6),
        (format_score, 0.1),
    ]
    n_hops = kwargs["info"]["n_hops"]
    difficulty_factor = n_hops / 2
    return difficulty_factor * sum(score * weight for score, weight in pairs) / sum(weight for _, weight in pairs)