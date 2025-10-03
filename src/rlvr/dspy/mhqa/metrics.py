from vf_musique.metrics import exact_match, f1


def metric_retrieval_recall(example, pred, trace=None):
    """Retrieval recall metric - fraction of supporting documents found."""
    if not example.supporting_ids:
        return 1.0  # No supporting documents to evaluate

    gold_ids = set(example.supporting_ids)
    retrieved_ids = set(pred.retrieved_doc_ids)

    if not gold_ids:
        return 1.0

    found = gold_ids.intersection(retrieved_ids)
    return len(found) / len(gold_ids)


def metric_retrieval_precision(example, pred, trace=None):
    """Retrieval precision metric - fraction of retrieved documents that are supporting."""
    if not example.supporting_ids:
        return 1.0

    gold_ids = set(example.supporting_ids)
    retrieved_ids = set(pred.retrieved_doc_ids)
    found = gold_ids.intersection(retrieved_ids)
    return len(found) / len(retrieved_ids)


def metric_answer_exact_match(example, pred, trace=None):
    """Exact match metric for MuSiQue using the official metrics."""
    return exact_match(pred.answer, example.answers)


def metric_answer_f1_score(example, pred, trace=None):
    """Token-level F1 score using the official metrics."""
    return f1(pred.answer, example.answers)


def metric_citation_f1(example, pred, trace=None):
    """Citation accuracy metrics - precision, recall, F1 for cited document IDs."""
    # Convert to sets for easy comparison
    gold_ids = set(example.supporting_ids) if example.supporting_ids else set()
    cited_ids = set(str(doc_id) for doc_id in pred.citations)  # Ensure string format

    # Handle edge cases
    if not gold_ids:
        raise ValueError("Supporting docs must be provided for citation metric")

    if not cited_ids:
        # No citations given but some needed
        return 0.0

    # Calculate standard precision/recall/F1
    correct_citations = cited_ids & gold_ids
    precision = len(correct_citations) / len(cited_ids)
    recall = len(correct_citations) / len(gold_ids)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return f1


def metric_n_hops_penalty(example, pred, trace=None):
    """N-hops penalty metric that penalizes agents taking more turns than reference."""
    # Get actual turns taken by the agent
    agent_turns = pred.n_turns

    # Get reference hops from the example
    reference_hops = example.n_hops

    if agent_turns <= reference_hops:
        # Perfect score if agent is efficient
        return 1.0
    else:
        # Exponential penalty for taking too many hops
        # Penalty = 0.8^(extra_hops)
        extra_hops = agent_turns - reference_hops
        return max(0.8**extra_hops, 0.1)  # Minimum score of 0.1


def metric(example, pred, trace=None):
    """Combined metric for MuSiQue: weighted by number of hops."""
    retrieval_recall_score = metric_retrieval_recall(example, pred, trace)
    retrieval_precision_score = metric_retrieval_precision(example, pred, trace)
    answer_f1_score = metric_answer_f1_score(example, pred, trace)
    citation_f1 = metric_citation_f1(example, pred, trace)
    n_hops_penalty_score = metric_n_hops_penalty(example, pred, trace)

    # Combine metrics: EM and F1 for answer quality, retrieval recall for completeness,
    # citation F1 for proper attribution, and hop efficiency penalty
    score_weight_pairs = [
        (retrieval_recall_score, 0.9),  # Retrieval recall for finding supporting docs
        (retrieval_precision_score, 0.5),  # Retrieval precision for finding supporting docs
        (answer_f1_score, 1.0),  # F1
        (citation_f1, 0.7),  # Citation accuracy for proper attribution
        (n_hops_penalty_score, 0.6),  # Hop efficiency penalty
    ]

    return sum(score * weight for score, weight in score_weight_pairs) / sum(weight for _, weight in score_weight_pairs)
