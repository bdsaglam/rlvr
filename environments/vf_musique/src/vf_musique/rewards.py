import re

from .metrics import exact_match, f1


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
    predicted_answer = parser.parse_answer(completion) or ""
    return exact_match(predicted_answer, info["answers"])


def f1_reward(completion, answer, info, parser, **kwargs):
    """F1 score reward function."""
    predicted_answer = parser.parse_answer(completion) or ""
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
    """Extract citations from the completion using XML parser."""
    assistant_messages = parser.get_assistant_messages(completion)
    if not assistant_messages:
        return []

    # Parse the content to extract citations
    parsed_response = parser.parse(assistant_messages[-1]["content"])
    if hasattr(parsed_response, cite_tag):
        cite_content = getattr(parsed_response, cite_tag)
        if cite_content:
            # Split by comma and clean up
            return [id.strip() for id in cite_content.split(",")]
    return []


def citation_metrics(completion, info, parser, **kwargs):
    """Citation accuracy metrics: precision, recall, F1 for cited documents."""
    cited_doc_ids = extract_citations(completion, parser, cite_tag="cite")
    supporting_doc_ids = [doc["id"] for doc in info["docs"] if doc["is_supporting"]]

    # Convert to sets for easy comparison
    cited_set = set(cited_doc_ids)
    supporting_set = set(supporting_doc_ids)

    # Handle edge cases
    if not supporting_set:
        # No supporting docs needed - perfect if no citations given, else penalized
        precision = 1.0 if not cited_set else 0.0
        recall = 1.0
        f1 = 1.0 if not cited_set else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    if not cited_set:
        # No citations given but some needed
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Calculate standard precision/recall/F1
    correct_citations = cited_set & supporting_set
    precision = len(correct_citations) / len(cited_set)
    recall = len(correct_citations) / len(supporting_set)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def citation_precision_reward(completion, info, parser, **kwargs):
    """Citation precision reward - fraction of cited documents that are supporting."""
    metrics = citation_metrics(completion, info, parser, **kwargs)
    return metrics["precision"]


def citation_recall_reward(completion, info, parser, **kwargs):
    """Citation recall reward - fraction of supporting documents that were cited."""
    metrics = citation_metrics(completion, info, parser, **kwargs)
    return metrics["recall"]


def citation_f1_reward(completion, info, parser, **kwargs):
    """Citation F1 reward - harmonic mean of citation precision and recall."""
    metrics = citation_metrics(completion, info, parser, **kwargs)
    return metrics["f1"]


def citation_reward(completion, info, parser, **kwargs):
    """Citation reward function based on verifiers citation.py pattern (backward compatibility)."""
    # Keep the old implementation for backward compatibility
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
    assistant_messages = parser.get_assistant_messages(completion)
    if not assistant_messages:
        return 0.0

    msg_content = assistant_messages[-1]["content"]

    # Qwen3 like thinking models don't include <think> tag in completion as it's automatically appended by the tokenizer
    if "</think>" in msg_content and not msg_content.strip().startswith("<think>"):
        msg_content = "<think>\n" + msg_content

    try:
        # Parse the content to check if it's well-formatted
        parsed_response = parser.parse(msg_content)

        score = 0.0
        tag_count = 0

        # Check if cite tag is present and parseable
        if hasattr(parsed_response, "cite"):
            cite_content = getattr(parsed_response, "cite")
            if cite_content is not None:
                score += 1.0
            tag_count += 1

        # Check if think tag is present and parseable
        if hasattr(parsed_response, "think"):
            think_content = getattr(parsed_response, "think")
            if think_content is not None:
                score += 1.0
            tag_count += 1

        # Check if answer tag is present and parseable
        if hasattr(parsed_response, "answer"):
            answer_content = getattr(parsed_response, "answer")
            if answer_content is not None:
                score += 1.0
            tag_count += 1

        # Return normalized score (0-1) based on successfully parsed tags
        return score / tag_count if tag_count > 0 else 0.0

    except Exception:
        # If parsing fails completely, return 0
        return 0.0


def combined_reward(*args, **kwargs):
    """Combined reward function."""
    # Get weighted scores, retrieval recall, and citation reward
    em_reward = exact_match_reward(*args, **kwargs)
    _f1_reward = f1_reward(*args, **kwargs)
    retrieval_recall = retrieval_recall_reward(*args, **kwargs)
    retrieval_precision = retrieval_precision_reward(*args, **kwargs)
    citation_f1_score = citation_f1_reward(*args, **kwargs)
    format_score = format_reward(*args, **kwargs)

    # Combine: average of weighted EM and F1, plus retrieval and citation bonuses
    pairs = [
        (em_reward, 1.0),  # Exact match is most important
        (_f1_reward, 0.8),  # F1 for partial credit
        (retrieval_recall, 0.6),  # Retrieval recall for finding supporting docs
        (retrieval_precision, 0.4),  # Retrieval precision for quality
        (citation_f1_score, 0.8),  # Citation F1 for proper attribution
        (format_score, 0.1),  # Format score for proper structure
    ]
    n_hops = kwargs["info"]["n_hops"]
    difficulty_factor = n_hops / 2
    return difficulty_factor * sum(score * weight for score, weight in pairs) / sum(weight for _, weight in pairs)
