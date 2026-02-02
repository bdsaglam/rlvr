"""ARC-AGI visual pattern reasoning environment for RLVR.

Trains LLMs to solve ARC-AGI puzzles using a persistent Python REPL
and an in-REPL submit_answer function.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

import verifiers as vf
from datasets import Dataset

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Grid = list[list[int]]

_SUBMISSION_FILE = "/tmp/arc_submission.json"


class ArcTaskInfo(TypedDict):
    task_id: str
    train_pairs: list[dict]  # [{"input": Grid, "output": Grid}, ...]
    test_inputs: list[Grid]
    test_outputs: list[Grid]
    num_test_cases: int
    grid_format: str  # "json" or "ascii"


# ---------------------------------------------------------------------------
# Grid formatting (pure functions)
# ---------------------------------------------------------------------------


def grid_to_json(grid: Grid) -> str:
    return json.dumps(grid)


def grid_to_ascii(grid: Grid) -> str:
    return "\n".join(" ".join("." if c == 0 else str(c) for c in row) for row in grid)


def format_grid(grid: Grid, fmt: str) -> str:
    if fmt == "ascii":
        return grid_to_ascii(grid)
    return grid_to_json(grid)


def format_task_question(
    train_pairs: list[dict],
    test_inputs: list[Grid],
    fmt: str,
) -> str:
    parts: list[str] = ["Training Examples:\n"]
    for i, pair in enumerate(train_pairs, 1):
        parts.append(f"Example {i}:")
        parts.append(f"Input:\n{format_grid(pair['input'], fmt)}")
        parts.append(f"Output:\n{format_grid(pair['output'], fmt)}\n")

    parts.append("Test Inputs:\n")
    for i, inp in enumerate(test_inputs, 1):
        parts.append(f"Test {i}:")
        parts.append(f"{format_grid(inp, fmt)}\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_arc_tasks(
    data_dir: str,
    split: str,
) -> tuple[dict, dict]:
    """Read challenges + solutions JSON files for the given split."""
    base = Path(data_dir)
    challenges_path = base / f"arc-agi_{split}_challenges.json"
    solutions_path = base / f"arc-agi_{split}_solutions.json"

    with open(challenges_path) as f:
        challenges = json.load(f)
    with open(solutions_path) as f:
        solutions = json.load(f)
    return challenges, solutions


def prepare_dataset(
    data_dir: str,
    split: str,
    grid_format: str = "json",
) -> Dataset:
    """Build a HuggingFace Dataset with one row per ARC task."""
    challenges, solutions = load_arc_tasks(data_dir, split)

    rows: list[dict] = []
    for task_id, task in challenges.items():
        train_pairs = task["train"]
        test_inputs = [t["input"] for t in task["test"]]
        test_outputs = solutions[task_id]  # list of grids

        question = format_task_question(train_pairs, test_inputs, grid_format)

        # answer: JSON string of expected outputs (for quick reference)
        answer = json.dumps(test_outputs)

        info = ArcTaskInfo(
            task_id=task_id,
            train_pairs=train_pairs,
            test_inputs=test_inputs,
            test_outputs=test_outputs,
            num_test_cases=len(test_outputs),
            grid_format=grid_format,
        )

        rows.append(
            {
                "question": question,
                "answer": answer,
                "info": json.dumps(info),
            }
        )

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------


def parse_submission(state) -> list[Grid] | None:
    """Extract submitted grids from state."""
    return state.get("submitted_answers")


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def exact_match_reward(state, info, **kwargs) -> float:
    """1.0 only if ALL predicted grids exactly match ALL expected grids."""
    grids = parse_submission(state)
    if grids is None:
        return 0.0
    expected = info["test_outputs"]
    if len(grids) != len(expected):
        return 0.0
    return 1.0 if all(g == e for g, e in zip(grids, expected)) else 0.0


def cell_accuracy_reward(state, info, **kwargs) -> float:
    """Average cell-level accuracy across all test cases."""
    grids = parse_submission(state)
    if grids is None:
        return 0.0
    expected = info["test_outputs"]
    if len(grids) != len(expected):
        return 0.0

    total_cells = 0
    correct_cells = 0
    for pred, exp in zip(grids, expected):
        if len(pred) != len(exp):
            # Shape mismatch for this test case — 0 accuracy
            total_cells += sum(len(row) for row in exp)
            continue
        for pred_row, exp_row in zip(pred, exp):
            if len(pred_row) != len(exp_row):
                total_cells += len(exp_row)
                continue
            for p, e in zip(pred_row, exp_row):
                total_cells += 1
                if p == e:
                    correct_cells += 1

    return correct_cells / total_cells if total_cells > 0 else 0.0


def shape_match_reward(state, info, **kwargs) -> float:
    """Fraction of test cases where predicted dimensions match expected."""
    grids = parse_submission(state)
    if grids is None:
        return 0.0
    expected = info["test_outputs"]
    if len(grids) != len(expected):
        return 0.0

    matches = 0
    for pred, exp in zip(grids, expected):
        if len(pred) == len(exp) and all(
            len(pr) == len(er) for pr, er in zip(pred, exp)
        ):
            matches += 1
    return matches / len(expected)


def format_reward(state, info, **kwargs) -> float:
    """1.0 if valid submission with correct number of grids."""
    grids = parse_submission(state)
    if grids is None:
        return 0.0
    expected = info["test_outputs"]
    return 1.0 if len(grids) == len(expected) else 0.0


# ---------------------------------------------------------------------------
# Rubric
# ---------------------------------------------------------------------------

_REWARD_MODE_WEIGHTS: dict[str, list[float]] = {
    # [exact_match, cell_accuracy, shape_match, format]
    "binary": [1.0, 0.0, 0.0, 0.0],
    "partial": [0.0, 1.0, 0.0, 0.0],
    "combined": [0.5, 0.5, 0.0, 0.0],
}


def ArcAgiRubric(parser=None, reward_mode: str = "binary", **kwargs):
    """Create an ARC-AGI rubric with configurable reward weighting."""
    funcs = [exact_match_reward, cell_accuracy_reward, shape_match_reward, format_reward]
    weights = _REWARD_MODE_WEIGHTS.get(reward_mode, _REWARD_MODE_WEIGHTS["binary"])
    return vf.Rubric(funcs=funcs, weights=weights, parser=parser, **kwargs)


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------


_REPL_SETUP_CODE = '''\
import numpy as np
import json

# --- ARC task data (auto-injected) ---
train_pairs = {train_pairs_json}
train_inputs = [np.array(p["input"]) for p in train_pairs]
train_outputs = [np.array(p["output"]) for p in train_pairs]
test_inputs = [np.array(t) for t in {test_inputs_json}]
num_test_cases = {num_test_cases}

# --- Utility functions ---
COLOR_NAMES = {{
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "pink", 7: "orange", 8: "cyan", 9: "maroon",
}}

def show(grid):
    """Pretty-print a grid (numpy array or list of lists)."""
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    for row in grid:
        print(" ".join(str(c) for c in row))

def grid_shape(grid):
    """Return (rows, cols) for a grid."""
    if hasattr(grid, "shape"):
        return grid.shape
    return (len(grid), len(grid[0]) if grid else 0)

def unique_colors(grid):
    """Return sorted list of unique color values in a grid."""
    if not hasattr(grid, "flatten"):
        grid = np.array(grid)
    return sorted(set(grid.flatten().tolist()))

def submit_answer(answers):
    """Submit final answer grids for all test inputs.

    Args:
        answers: List of output grids, one per test input.
                 Each grid can be a numpy array or a list of lists.
                 Values must be integers 0-9.

    Example:
        result = transform(test_inputs[0])
        submit_answer([result])
    """
    grids = []
    for grid in answers:
        if hasattr(grid, "tolist"):
            grid = grid.tolist()
        grids.append([[int(c) for c in row] for row in grid])
    with open("{submission_file}", "w") as f:
        json.dump(grids, f)
    print("Answers submitted.")
'''


class ArcAgiEnv(vf.PythonEnv):
    def __init__(self, **kwargs):
        super().__init__(pip_install_packages="numpy", **kwargs)

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        state["submitted_answers"] = None
        info = state["info"]
        setup_code = _REPL_SETUP_CODE.format(
            train_pairs_json=json.dumps(info["train_pairs"]),
            test_inputs_json=json.dumps(info["test_inputs"]),
            num_test_cases=info["num_test_cases"],
            submission_file=_SUBMISSION_FILE,
        )
        await self.python(
            code=setup_code,
            sandbox_id=state["sandbox_id"],
            sandbox_state=state["sandbox_state"],
            python_state=state["python_state"],
        )
        return state

    async def _check_submission(self, state: vf.State) -> bool:
        """Check if submit_answer was called in the REPL and read the result."""
        if state.get("submitted_answers") is not None:
            return True
        sandbox_id = state["sandbox_id"]
        sandbox_state = state["sandbox_state"]
        check_cmd = f'cat {_SUBMISSION_FILE} 2>/dev/null || echo "__NO_SUBMISSION__"'
        result = await self.bash(check_cmd, sandbox_id, sandbox_state)
        result = result.strip()
        if result and result != "__NO_SUBMISSION__":
            try:
                state["submitted_answers"] = json.loads(result)
                return True
            except json.JSONDecodeError:
                pass
        return False

    @vf.stop
    async def answer_submitted(self, state: vf.State) -> bool:
        """Stop when submit_answer is called in the REPL."""
        return await self._check_submission(state)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_FORMAT_NOTES = {
    "json": "Grids are formatted as JSON arrays (e.g. [[1, 0], [0, 1]]).",
    "ascii": (
        "Grids are formatted as ASCII art where '.' represents 0 (black) "
        "and digits 1-9 represent colors. Columns are space-separated."
    ),
}

_SYSTEM_PROMPT = """\
You are solving an ARC-AGI puzzle. You are given training examples (input/output
grid pairs) that demonstrate a transformation pattern. Your task is to discover
the pattern and apply it to the test input(s) to produce the correct output grid(s).

Grids are 2D arrays of integers 0-9 representing colors:
  0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=pink 7=orange 8=cyan 9=maroon

{format_note}

The Python environment has these pre-loaded variables and utilities:
- `train_pairs`: list of {{"input": grid, "output": grid}} dicts
- `train_inputs`: list of numpy arrays (one per training input)
- `train_outputs`: list of numpy arrays (one per training output)
- `test_inputs`: list of numpy arrays (one per test input)
- `num_test_cases`: number of test inputs to solve
- `show(grid)`: pretty-print a grid
- `grid_shape(grid)`: return (rows, cols)
- `unique_colors(grid)`: return sorted unique color values
- `np` (numpy) is imported

Strategy:
1. Study the training examples carefully. Look for patterns in how inputs transform
   to outputs: spatial transforms, color mappings, object detection, symmetry,
   counting, filling, etc.
2. Use the `python` tool to write and execute code that helps you analyze patterns,
   test hypotheses, or generate the output grid(s). Variables persist between calls.
3. When confident, call `submit_answer(answers)` inside the Python REPL where
   `answers` is a list of output grids (one per test input, in order). Each grid
   can be a numpy array or a list of lists.

Example submission:
```
result = transform(test_inputs[0])
submit_answer([result])
```

Rules:
- Call `submit_answer` exactly once with your final answers for ALL test inputs.
- Each answer grid must contain integers 0-9.
- You can only submit answers by calling `submit_answer()` inside the Python REPL."""


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def load_environment(
    data_dir: str = "data/arc-prize-2025",
    split: str = "training",
    eval_data_dir: str | None = None,
    eval_split: str = "evaluation",
    grid_format: str = "json",
    reward_mode: str = "binary",
    max_turns: int = 10,
    **kwargs,
) -> vf.Environment:
    """Load the ARC-AGI environment."""
    dataset = prepare_dataset(data_dir, split, grid_format=grid_format)

    eval_dataset = None
    if eval_data_dir is not None:
        eval_dataset = prepare_dataset(eval_data_dir, eval_split, grid_format=grid_format)

    format_note = _FORMAT_NOTES.get(grid_format, _FORMAT_NOTES["json"])
    system_prompt = _SYSTEM_PROMPT.format(format_note=format_note)

    parser = vf.Parser()
    rubric = ArcAgiRubric(parser=parser, reward_mode=reward_mode)

    env = ArcAgiEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_turns=max_turns,
        **kwargs,
    )

    return env
