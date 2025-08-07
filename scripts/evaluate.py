import json
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from verifiers.metrics.musique import exact_match, f1
from verifiers.models import Message
from verifiers.rubrics.citation import make_citation_extractor
from verifiers.rubrics.utils import get_last_answer
from verifiers.tools.retrieve import extract_all_retrieved_doc_ids


def pick_trajectory(group: pd.DataFrame) -> pd.Series:
    """Pick a row from the group based on majority voting on `predicted_answer`."""
    # Get value counts for 'predicted_answer'. By default, NaN values are dropped.
    predicted_answers_counts = group["predicted_answer"].value_counts()

    if predicted_answers_counts.empty:
        # This occurs if all 'predicted_answer' values in the group are NaN.
        # Since the group itself is guaranteed not to be empty, we fallback
        # to returning the first row of the group.
        return group.iloc[0]
    else:
        majority_answer = predicted_answers_counts.idxmax()
        # Select the first row that has this majority answer.
        return group[group["predicted_answer"] == majority_answer].iloc[0]


citation_extractor = make_citation_extractor()


def calculate_citation_metrics(trajectory: list[Message], docs: list[dict]) -> dict[str, float]:
    """Calculate citation metrics."""
    cited_doc_ids = citation_extractor(trajectory)
    if not cited_doc_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    supporting_doc_ids = [doc["id"] for doc in docs if doc["is_supporting"]]
    if not supporting_doc_ids:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = len(set(cited_doc_ids) & set(supporting_doc_ids))
    precision = intersection / len(cited_doc_ids) if cited_doc_ids else 0.0
    recall = intersection / len(supporting_doc_ids) if supporting_doc_ids else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1_score}


def calculate_supporting_metrics(retrieved: set[str], supporting: set[str]) -> dict[str, float]:
    """Calculate precision, recall, and F1 for supporting titles.

    Args:
        retrieved: Set of retrieved titles
        supporting: Set of supporting titles

    Returns:
        Dictionary containing precision, recall, and F1 scores
    """
    if not retrieved:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if not supporting:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = len(retrieved & supporting)
    precision = intersection / len(retrieved) if retrieved else 0.0
    recall = intersection / len(supporting) if supporting else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1_score}


def process_row(row: pd.Series) -> dict[str, Any]:
    """Process a single row of data to compute all metrics.

    Args:
        row: DataFrame row containing trajectory, supporting_titles, and answers

    Returns:
        Dictionary containing all computed metrics
    """
    # Extract data
    predicted_answer = get_last_answer(row["trajectory"])
    supporting_doc_ids = set(doc["id"] for doc in row["docs"] if doc["is_supporting"])
    retrieved_doc_ids = set(extract_all_retrieved_doc_ids(row["trajectory"]))

    # Calculate answer metrics
    answer_exact_match = exact_match(predicted_answer, row["answers"])
    answer_f1 = f1(predicted_answer, row["answers"])

    # Calculate supporting metrics
    supporting_metrics = calculate_supporting_metrics(retrieved_doc_ids, supporting_doc_ids)

    # Calculate citation metrics
    citation_metrics = calculate_citation_metrics(row["trajectory"], row["docs"])

    return {
        "predicted_answer": predicted_answer,
        "exact_match": answer_exact_match,
        "f1": answer_f1,
        "supporting_doc_ids": list(supporting_doc_ids),
        "retrieved_doc_ids": list(retrieved_doc_ids),
        "supporting.precision": supporting_metrics["precision"],
        "supporting.recall": supporting_metrics["recall"],
        "supporting.f1": supporting_metrics["f1"],
        "citation.precision": citation_metrics["precision"],
        "citation.recall": citation_metrics["recall"],
        "citation.f1": citation_metrics["f1"],
    }


def evaluate(filepath: Path = typer.Argument(), out: Path = typer.Option(...)):
    """Evaluate model performance and save results.

    Args:
        filepath: Path to the input JSON Lines file
        output_dir: Directory to save output files
    """
    # Create output directory if it doesn't exist
    out.mkdir(parents=True, exist_ok=True)

    # Load data
    preds_df = pd.read_json(filepath, lines=True)
    preds_df["predicted_answer"] = preds_df["trajectory"].apply(get_last_answer)

    # Pick a trajectory for each group based on majority voting on `predicted_answer`.
    agg_preds_df = preds_df.groupby("id").apply(pick_trajectory).reset_index(drop=True)

    # Process all metrics in a single pass
    metrics_df = pd.DataFrame(agg_preds_df.apply(process_row, axis=1).tolist())
    result_df = pd.concat(
        [agg_preds_df[["id", "n_hops", "answers", "supporting_doc_slugs"]], metrics_df],
        axis=1,
    )

    # Select columns for output
    columns = [
        "id",
        "n_hops",
        # QA
        "answers",
        "predicted_answer",
        "exact_match",
        "f1",
        # Retrieval
        "supporting_doc_ids",
        "retrieved_doc_ids",
        "supporting.precision",
        "supporting.recall",
        "supporting.f1",
        "citation.precision",
        "citation.recall",
        "citation.f1",
    ]

    # Save results
    result_df[columns].to_json(out / "results.jsonl", orient="records", lines=True)

    # Calculate and save aggregate scores
    score_columns = [
        "exact_match",
        "f1",
        "supporting.precision",
        "supporting.recall",
        "supporting.f1",
        "citation.precision",
        "citation.recall",
        "citation.f1",
    ]
    scores = result_df[score_columns].mean().to_dict()

    with open(out / "scores.json", "w") as f:
        json.dump(scores, f, indent=2)


if __name__ == "__main__":
    typer.run(evaluate)
