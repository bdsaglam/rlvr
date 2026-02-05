"""Data loading and formatting for ARC-AGI environments."""

from __future__ import annotations

import json
import re
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
# Split parsing
# ---------------------------------------------------------------------------


def parse_split(split_str: str) -> tuple[str, int | None, int | None, list[str] | None]:
    """Parse split string with optional range or task ID notation.

    Supports formats:
    - "training" -> ("training", None, None, None)
    - "training[:100]" -> ("training", None, 100, None)
    - "training[50:]" -> ("training", 50, None, None)
    - "training[50:100]" -> ("training", 50, 100, None)
    - "training[[abc123,def456]]" -> ("training", None, None, ["abc123", "def456"])

    Args:
        split_str: Split string with optional range or task IDs.

    Returns:
        Tuple of (split_name, start, end, task_ids).
    """
    # Try task ID pattern first: split[[id1,id2,id3]]
    match = re.match(r"^(\w+)\[\[([^\]]+)\]\]$", split_str)
    if match:
        split_name = match.group(1)
        task_ids = [p.strip() for p in match.group(2).split(",")]
        return split_name, None, None, task_ids

    # Try range pattern: split[start:end]
    match = re.match(r"^(\w+)(?:\[(\d*):(\d*)\])?$", split_str)
    if not match:
        return split_str, None, None, None

    split_name = match.group(1)
    start_str = match.group(2)
    end_str = match.group(3)

    start = int(start_str) if start_str else None
    end = int(end_str) if end_str else None

    return split_name, start, end, None


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
    """Build a HuggingFace Dataset with one row per ARC task.

    Args:
        data_dir: Path to ARC data directory.
        split: Split string with optional range or task IDs.
            Examples:
            - "training" - all tasks
            - "training[:100]" - first 100 tasks
            - "training[50:100]" - tasks 50-100
            - "training[[abc123,def456]]" - specific task IDs

    Returns:
        HuggingFace Dataset with columns: question, answer, info.
    """
    # Parse split string for range/task filtering
    split_name, start, end, task_ids = parse_split(split)

    challenges, solutions = load_arc_tasks(data_dir, split_name)

    # Filter by task IDs if specified
    if task_ids:
        challenges = {k: v for k, v in challenges.items() if k in task_ids}
        solutions = {k: v for k, v in solutions.items() if k in task_ids}

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

    # Apply range slicing if specified
    if start is not None or end is not None:
        rows = rows[start:end]

    return Dataset.from_list(rows)
