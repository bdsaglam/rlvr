"""Data loading and formatting for ARC-AGI environments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from datasets import Dataset

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Grid = list[list[int]]


class ArcTaskInfo(TypedDict):
    task_id: str
    train_pairs: list[dict]  # [{"input": Grid, "output": Grid}, ...]
    test_inputs: list[Grid]
    test_outputs: list[Grid]
    num_test_cases: int


# ---------------------------------------------------------------------------
# Grid formatting
# ---------------------------------------------------------------------------


def grid_to_diagram(grid: Grid) -> str:
    """Convert a grid to an ASCII diagram (space-separated digits)."""
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def format_input_output_pair(input: Grid, output: Grid) -> str:
    return f"Input:\n<Diagram>\n{grid_to_diagram(input)}\n</Diagram>\nOutput:\n<Diagram>\n{grid_to_diagram(output)}\n</Diagram>"


def format_pairs(pairs: list[dict], split: str = "train") -> str:
    parts: list[str] = []
    title = "Example" if split == "train" else "Challenge"
    for i, pair in enumerate(pairs, 1):
        parts.append(f"{title} #{i}")
        parts.append(format_input_output_pair(pair["input"], pair["output"]))
    return "\n".join(parts)


def format_task_question(train_pairs: list[dict], test_inputs: list[Grid]) -> str:
    """Format an ARC task as a question string with diagrams."""
    parts: list[str] = []
    for i, pair in enumerate(train_pairs, 1):
        parts.append(f"Example #{i}")
        parts.append("Input:")
        parts.append(f"<Diagram>\n{grid_to_diagram(pair['input'])}\n</Diagram>")
        parts.append("Output:")
        parts.append(f"<Diagram>\n{grid_to_diagram(pair['output'])}\n</Diagram>")
        parts.append("")

    for i, inp in enumerate(test_inputs, 1):
        parts.append(f"Challenge #{i}")
        parts.append("Input:")
        parts.append(f"<Diagram>\n{grid_to_diagram(inp)}\n</Diagram>")
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_arc_tasks(data_dir: str, split: str) -> tuple[dict, dict]:
    """Read challenges + solutions JSON files for the given split."""
    base = Path(data_dir)
    challenges_path = base / f"arc-agi_{split}_challenges.json"
    solutions_path = base / f"arc-agi_{split}_solutions.json"

    with open(challenges_path) as f:
        challenges = json.load(f)
    with open(solutions_path) as f:
        solutions = json.load(f)
    return challenges, solutions


def prepare_dataset(data_dir: str, split: str) -> Dataset:
    """Build a HuggingFace Dataset with one row per ARC task."""
    challenges, solutions = load_arc_tasks(data_dir, split)

    rows: list[dict] = []
    for task_id, task in challenges.items():
        train_pairs = task["train"]
        test_inputs = [t["input"] for t in task["test"]]
        test_outputs = solutions[task_id]

        question = format_task_question(train_pairs, test_inputs)
        answer = json.dumps(test_outputs)

        info = ArcTaskInfo(
            task_id=task_id,
            train_pairs=train_pairs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            num_test_cases=len(test_outputs),
        )

        rows.append(
            {
                "question": question,
                "answer": answer,
                "info": json.dumps(info),
            }
        )

    return Dataset.from_list(rows)
