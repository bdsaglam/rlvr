from .metrics import exact_match, extract_all_retrieved_doc_ids, f1


# Reward functions for MuSiQue evaluation
def exact_match_reward(completion, answer, info, parser, **kwargs):
    """Exact match reward function."""
    predicted_answer = parser.parse_answer(completion) or ""
    return exact_match(predicted_answer, info["answers"])


def f1_reward(completion, answer, info, parser, **kwargs):
    """F1 score reward function."""
    predicted_answer = parser.parse_answer(completion) or ""
    return f1(predicted_answer, info["answers"])


def weighted_exact_match_reward(completion, answer, info, parser, **kwargs):
    """Weighted exact match reward (harder for multi-hop questions)."""
    predicted_answer = parser.parse_answer(completion) or ""
    em_score = exact_match(predicted_answer, info["answers"])
    n_hops = info.get("n_hops", 1)
    # Weight decreases as number of hops increases (harder questions)
    weight = 1.0 / n_hops if n_hops > 0 else 1.0
    return em_score * weight


def weighted_f1_reward(completion, answer, info, parser, **kwargs):
    """Weighted F1 reward (harder for multi-hop questions)."""
    predicted_answer = parser.parse_answer(completion) or ""
    f1_score = f1(predicted_answer, info["answers"])
    n_hops = info.get("n_hops", 1)
    # Weight decreases as number of hops increases (harder questions)
    weight = 1.0 / n_hops if n_hops > 0 else 1.0
    return f1_score * weight


def retrieval_recall_reward(completion, info, **kwargs):
    """Retrieval recall reward function."""
    supporting_doc_ids = [doc["id"] for doc in info["docs"] if doc["is_supporting"]]
    retrieved_doc_ids = set(extract_all_retrieved_doc_ids(completion))

    if len(retrieved_doc_ids) > 0 and len(supporting_doc_ids) > 0:
        return len(retrieved_doc_ids & set(supporting_doc_ids)) / len(supporting_doc_ids)
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
    assistant_messages = parser.get_assistant_messages(completion)
    if not assistant_messages:
        return 0.0

    try:
        # Parse the content to check if it's well-formatted
        parsed_response = parser.parse(assistant_messages[-1]["content"])

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
    weighted_em = weighted_exact_match_reward(*args, **kwargs)
    weighted_f1 = weighted_f1_reward(*args, **kwargs)
    retrieval_recall = retrieval_recall_reward(*args, **kwargs)
    citation_score = citation_reward(*args, **kwargs)
    format_score = format_reward(*args, **kwargs)

    # Combine: average of weighted EM and F1, plus retrieval and citation bonuses
    pairs = [
        (weighted_em, 1),
        (weighted_f1, 0.9),
        (citation_score, 0.6),
        (retrieval_recall, 0.4),
        (format_score, 0.1),
    ]
    return sum(score * weight for score, weight in pairs) / sum(weight for _, weight in pairs)
