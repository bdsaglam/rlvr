import dspy
from vf_musique.metrics import exact_match, f1

from .metrics import metric_n_hops_penalty


def feedback_retrieval_recall(example, pred):
    """Generate feedback for retrieval recall evaluation."""
    gold_ids = set(example.supporting_ids)
    retrieved_ids = set(pred.retrieved_doc_ids)
    found = gold_ids.intersection(retrieved_ids)
    recall_score = len(found) / len(gold_ids)

    if recall_score == 1.0:
        feedback = f"Perfect retrieval! You found all {len(gold_ids)} supporting documents: {sorted(found)}"
    elif recall_score >= 0.5:
        missing_ids = gold_ids - found
        feedback = (
            f"Good retrieval (recall: {recall_score:.2f}). Found {len(found)} out of {len(gold_ids)} "
            f"supporting documents. Missing: {sorted(missing_ids)}. Consider refining your search queries "
            f"to find the remaining relevant documents."
        )
    else:
        missing_ids = gold_ids - found
        feedback = (
            f"Poor retrieval (recall: {recall_score:.2f}). Only found {len(found)} out of {len(gold_ids)} "
            f"supporting documents. Missing critical documents: {sorted(missing_ids)}. "
            f"Your search queries need to be more comprehensive and targeted."
        )

    return recall_score, feedback


def feedback_retrieval_precision(example, pred):
    """Generate feedback for retrieval precision evaluation."""
    gold_ids = set(example.supporting_ids)
    retrieved_ids = set(pred.retrieved_doc_ids)

    if not retrieved_ids:
        return 0.0, "No documents were retrieved. Your search queries need to find relevant documents."

    found = gold_ids.intersection(retrieved_ids)
    precision_score = len(found) / len(retrieved_ids)
    irrelevant_docs = retrieved_ids - gold_ids

    if precision_score == 1.0:
        feedback = (
            f"Perfect precision! All {len(retrieved_ids)} retrieved documents are supporting documents: {sorted(found)}"
        )
    elif precision_score >= 0.7:
        feedback = (
            f"Good precision (precision: {precision_score:.2f}). {len(found)} out of {len(retrieved_ids)} "
            f"retrieved documents are relevant. Irrelevant docs: {sorted(irrelevant_docs)}. "
            f"Consider making your search queries more specific to avoid irrelevant documents."
        )
    elif precision_score >= 0.3:
        feedback = (
            f"Moderate precision (precision: {precision_score:.2f}). Only {len(found)} out of {len(retrieved_ids)} "
            f"retrieved documents are relevant. Many irrelevant docs retrieved: {sorted(irrelevant_docs)}. "
            f"Your search queries are too broad - focus on more specific terms and entities."
        )
    else:
        feedback = (
            f"Poor precision (precision: {precision_score:.2f}). Only {len(found)} out of {len(retrieved_ids)} "
            f"retrieved documents are relevant. Most retrieved docs are irrelevant: {sorted(irrelevant_docs)}. "
            f"Your search queries are retrieving too many irrelevant documents. Be much more specific and targeted."
        )

    return precision_score, feedback


def feedback_answer_exact_match(example, pred):
    """Generate feedback for exact match evaluation."""
    em_score = exact_match(pred.answer, example.answers)

    if em_score == 1.0:
        feedback = f"Perfect! You provided the exact correct answer: '{pred.answer}'. This matches the expected answer exactly."
    else:
        # Find the best matching answer for more specific feedback
        best_answer = example.answers[0] if example.answers else "N/A"
        feedback = (
            f"Your answer '{pred.answer}' doesn't exactly match the expected answer '{best_answer}'. "
            f"Consider being more precise with entity names, dates, and specific facts."
        )

    return em_score, feedback


def feedback_answer_f1_score(example, pred):
    """Generate feedback for F1 score evaluation."""
    f1_score = f1(pred.answer, example.answers)

    if f1_score >= 0.9:
        feedback = f"Excellent! Your answer has high overlap (F1: {f1_score:.2f}) with the expected answer. Good token-level accuracy."
    elif f1_score >= 0.5:
        feedback = (
            f"Good partial match (F1: {f1_score:.2f}). Your answer contains relevant information but "
            f"could be more complete or precise. Consider including more specific details from the retrieved documents."
        )
    else:
        best_answer = example.answers[0] if example.answers else "N/A"
        feedback = (
            f"Low overlap (F1: {f1_score:.2f}) with expected answer. Your answer '{pred.answer}' "
            f"differs significantly from '{best_answer}'. Focus on extracting the specific information "
            f"requested in the question."
        )

    return f1_score, feedback


def feedback_citation_f1(example, pred):
    """Generate feedback for citation F1 evaluation."""
    gold_ids = set(example.supporting_ids) if example.supporting_ids else set()
    cited_ids = set(str(doc_id) for doc_id in pred.citations)

    if not gold_ids:
        return 1.0, "No supporting documents to cite."

    if not cited_ids:
        feedback = f"You didn't cite any documents, but should have cited: {sorted(gold_ids)}. Always cite the documents that support your answer."
        return 0.0, feedback

    correct_citations = cited_ids & gold_ids
    precision = len(correct_citations) / len(cited_ids)
    recall = len(correct_citations) / len(gold_ids)
    citation_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    if citation_f1 >= 0.9:
        feedback = f"Excellent citations (F1: {citation_f1:.2f})! You properly cited the supporting documents: {sorted(correct_citations)}"
    elif citation_f1 >= 0.5:
        incorrect_citations = cited_ids - gold_ids
        missing_citations = gold_ids - cited_ids
        feedback = f"Good citations (F1: {citation_f1:.2f}). Correct: {sorted(correct_citations)}. "
        if incorrect_citations:
            feedback += f"Unnecessary: {sorted(incorrect_citations)}. "
        if missing_citations:
            feedback += f"Missing: {sorted(missing_citations)}. "
        feedback += "Be more precise about which documents actually support your answer."
    else:
        incorrect_citations = cited_ids - gold_ids
        missing_citations = gold_ids - cited_ids
        feedback = (
            f"Poor citations (F1: {citation_f1:.2f}). You cited {sorted(cited_ids)} but should cite {sorted(gold_ids)}. "
            f"Focus on identifying which documents directly support your answer claims."
        )

    return citation_f1, feedback


def feedback_n_hops_penalty(example, pred):
    """Generate feedback for n-hops penalty evaluation."""
    # Get actual turns taken by the agent
    agent_turns = pred.n_turns

    # Get reference hops from the example
    reference_hops = example.n_hops

    penalty_score = metric_n_hops_penalty(example, pred)

    # Analyze retrieval patterns to provide specific feedback
    retrieved_docs = pred.retrieved_doc_ids
    unique_docs = len(set(retrieved_docs))
    total_retrievals = len(retrieved_docs)

    if agent_turns <= reference_hops:
        feedback = f"Perfect efficiency! You completed the task in {agent_turns} turns, same as or fewer than the reference ({reference_hops} hops). This shows excellent retrieval strategy and reasoning efficiency."
    elif agent_turns == reference_hops + 1:
        if total_retrievals > unique_docs:
            feedback = f"Good efficiency (penalty: {penalty_score:.2f}). You took {agent_turns} turns vs reference {reference_hops} hops. Only 1 extra turn - but check if you retrieved duplicate documents ({total_retrievals} retrievals, {unique_docs} unique). Focus on more targeted initial queries."
        else:
            feedback = f"Good efficiency (penalty: {penalty_score:.2f}). You took {agent_turns} turns vs reference {reference_hops} hops. Only 1 extra turn - consider if your initial retrieval query could have been more comprehensive to get the needed information upfront."
    elif agent_turns <= reference_hops + 2:
        if total_retrievals > unique_docs:
            redundant_retrievals = total_retrievals - unique_docs
            feedback = f"Moderate efficiency (penalty: {penalty_score:.2f}). You took {agent_turns} turns vs reference {reference_hops} hops. You made {redundant_retrievals} redundant retrieval(s) - avoid retrieving the same documents multiple times. Plan your queries more strategically."
        else:
            feedback = f"Moderate efficiency (penalty: {penalty_score:.2f}). You took {agent_turns} turns vs reference {reference_hops} hops. Your retrieval queries may be too specific or missing key entities. Try broader, more comprehensive initial searches."
    else:
        extra_turns = agent_turns - reference_hops
        if total_retrievals > unique_docs:
            redundant_retrievals = total_retrievals - unique_docs
            feedback = f"Poor efficiency (penalty: {penalty_score:.2f}). You took {extra_turns} extra turns ({agent_turns} vs {reference_hops} reference). Major issue: {redundant_retrievals} redundant retrieval(s) - you're wasting turns retrieving documents you already have. Focus on tracking what you've retrieved and crafting better initial queries."
        else:
            feedback = f"Poor efficiency (penalty: {penalty_score:.2f}). You took {extra_turns} extra turns ({agent_turns} vs {reference_hops} reference). Your retrieval strategy is inefficient - queries may be too narrow or poorly targeted. Plan what information you need and retrieve it strategically in fewer, more comprehensive searches."

    return penalty_score, feedback


def metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Combined metric for MuSiQue with feedback for GEPA optimization.
    Returns a dspy.Prediction with score (float) and feedback (str).

    The feedback is targeted at specific predictors when pred_name is provided,
    helping GEPA understand how to improve each component.
    """
    # Compute feedback and scores for all metrics
    score_answer_f1, fb_answer_f1 = feedback_answer_f1_score(example, pred)
    score_retrieval_recall, fb_retrieval_recall = feedback_retrieval_recall(example, pred)
    score_retrieval_precision, fb_retrieval_precision = feedback_retrieval_precision(example, pred)
    score_citation_f1, fb_citation_f1 = feedback_citation_f1(example, pred)
    score_n_hops_penalty, fb_n_hops_penalty = feedback_n_hops_penalty(example, pred)

    # Combined score: weighted average of all metrics (same as original metric)
    score_weight_pairs = [
        (score_answer_f1, 1.0),  # Answer F1
        (score_retrieval_recall, 0.9),  # Retrieval recall for finding supporting docs
        (score_retrieval_precision, 0.5),  # Retrieval precision for finding supporting docs
        (score_citation_f1, 0.7),  # Citation accuracy for proper attribution
        (score_n_hops_penalty, 0.6),  # Hop efficiency penalty
    ]

    total_score = sum(score * weight for score, weight in score_weight_pairs) / sum(
        weight for _, weight in score_weight_pairs
    )

    # Provide targeted feedback based on the predictor being optimized
    if pred_name == "generate_query.predict":
        # Focus on query generation quality and retrieval effectiveness
        feedback = (
            fb_retrieval_recall
            + " "
            + fb_retrieval_precision
            + " "
            + fb_n_hops_penalty
            + " "
            + "Your search queries should be both comprehensive (high recall) and specific (high precision). "
            "Consider what entities, relationships, or facts are needed for each hop of reasoning. "
            "Plan your queries strategically to avoid unnecessary turns."
        )

    elif pred_name == "extract_info.predict":
        # Focus on information extraction quality
        feedback = (
            fb_answer_f1
            + " "
            + (
                "Focus on extracting the most relevant facts, entities, and relationships from the retrieved documents. "
                "Make sure to capture information that directly helps answer the question or leads to the next reasoning step."
            )
        )

    elif pred_name == "generate_answer.predict":
        # Focus on answer generation and citation quality
        feedback = (
            fb_answer_f1
            + " "
            + fb_citation_f1
            + " "
            + (
                "Provide precise, complete answers using the exact information from the retrieved documents. "
                "Always cite the document IDs that support your answer claims."
            )
        )
    else:
        # Generic feedback combining all aspects
        feedback = "\n".join(
            [
                "Overall performance breakdown:",
                f"- Answer F1 Score: {fb_answer_f1}",
                f"- Retrieval Recall: {fb_retrieval_recall}",
                f"- Retrieval Precision: {fb_retrieval_precision}",
                f"- Citations F1 Score: {fb_citation_f1}",
                f"- Hop Efficiency: {fb_n_hops_penalty}",
            ]
        )

    return dspy.Prediction(score=total_score, feedback=feedback)
