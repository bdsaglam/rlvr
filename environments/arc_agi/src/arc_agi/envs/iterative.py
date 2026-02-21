"""Iterative refinement environment for ARC-AGI.

The LLM writes a `transform` function in markdown code blocks.
The function is evaluated on training examples in a subprocess sandbox:
- If all pass: task complete, function applied to test inputs
- If not: provide feedback and continue
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import verifiers as vf

from ..sandbox import evaluate_on_train, execute_transform

# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_python_code(text: str) -> str | None:
    """Extract the last Python code block from markdown text."""
    matches = _CODE_BLOCK_PATTERN.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


# ---------------------------------------------------------------------------
# Feedback generation
# ---------------------------------------------------------------------------


def build_feedback(results: list[dict]) -> str:
    """Build feedback string from evaluation results."""
    parts = []
    num_passed = sum(1 for r in results if r["passed"])
    total = len(results)

    parts.append(f"Evaluation: {num_passed}/{total} training examples passed.\n")

    for r in results:
        i = r["index"] + 1
        if r["passed"]:
            parts.append(f"Example #{i}: PASS")
        elif r["error"]:
            parts.append(f"Example #{i}: FAIL - {r['error']}")
        else:
            parts.append(f"Example #{i}: FAIL - Accuracy: {r['accuracy']:.1%}")
            pred_arr = np.array(r["predicted"])
            exp_arr = np.array(r["expected"])
            diff_lines = []
            for row_i in range(min(pred_arr.shape[0], 15)):
                row_parts = []
                for col_i in range(min(pred_arr.shape[1], 15)):
                    p, e = pred_arr[row_i, col_i], exp_arr[row_i, col_i]
                    if p == e:
                        row_parts.append(f"{p:2}")
                    else:
                        row_parts.append(f"{p}/{e}")
                diff_lines.append(" ".join(row_parts))
            parts.append("  Diff (predicted/expected):")
            for line in diff_lines:
                parts.append(f"    {line}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are solving an ARC-AGI puzzle. You are given training examples (input/output
grid pairs) that demonstrate a transformation pattern. Your task is to discover
the pattern and write a Python function that implements it.

Grids are 2D arrays of integers 0-9 representing colors:
  0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=pink 7=orange 8=cyan 9=maroon

Grids are shown as space-separated digits inside <Diagram> tags.

**Your task:**
Write a Python function called `transform(grid: np.ndarray) -> np.ndarray` that
correctly transforms any input grid to its corresponding output.

**Instructions:**
1. Analyze the training examples carefully
2. Identify the transformation pattern
3. Write a `transform` function in a ```python code block
4. Your code will be automatically tested on all training examples
5. If any example fails, you'll receive feedback - refine your solution
6. Once all training examples pass, your solution is submitted

**Code requirements:**
- Function must be named `transform`
- Input: numpy array (2D grid of ints 0-9)
- Output: numpy array (2D grid of ints 0-9)
- numpy is available as `np`

**Example:**
```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    \"\"\"Rotate the grid 90 degrees clockwise.\"\"\"
    return np.rot90(grid, k=-1)
```

Focus on understanding the pattern from the examples. Start simple and refine
based on feedback."""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ArcAgiIterativeEnv(vf.MultiTurnEnv):
    """Iterative refinement environment for ARC-AGI.

    The LLM writes a `transform` function in markdown code blocks.
    The function is evaluated on training examples:
    - If all pass: task complete, apply to test inputs
    - If not: provide feedback and continue
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, timeout_s: float = 2.0, **kwargs):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.timeout_s = timeout_s

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        state["submitted_answers"] = None
        state["best_code"] = None
        state["best_score"] = -1.0
        state["iteration"] = 0
        return state

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        """Extract transform function, evaluate, and provide feedback."""
        state["iteration"] += 1

        # Find the last assistant message
        assistant_msg = None
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                assistant_msg = msg
                break

        if assistant_msg is None:
            return []

        content = assistant_msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )

        # Extract code
        code = extract_python_code(str(content) if content else "")
        if not code:
            return [{"role": "user", "content": "No Python code block found. Please provide a `transform` function in a ```python code block."}]

        # Get task info
        info = state["info"]

        # Evaluate on training examples using sandbox
        results = evaluate_on_train(code, info["train"], timeout_s=self.timeout_s)
        num_passed = sum(1 for r in results if r["passed"])
        total = len(results)
        avg_accuracy = sum(r["accuracy"] for r in results) / total if total > 0 else 0.0

        # Track best solution
        if avg_accuracy > state["best_score"]:
            state["best_score"] = avg_accuracy
            state["best_code"] = code

        # Check if all passed
        if num_passed == total:
            # Success! Apply to test inputs and submit
            train_preds = [r["predicted"] for r in results]
            test_preds = []
            for test_example in info["test"]:
                pred, _ = execute_transform(code, test_example["input"], timeout_s=self.timeout_s)
                test_preds.append(pred if pred is not None else [])

            state["submitted_answers"] = {
                "train": train_preds,
                "test": test_preds,
            }

            # Return success message and stop
            state["final_env_response"] = [
                {
                    "role": "user",
                    "content": f"All {total} training examples passed! Your solution has been submitted.\n\nFinal code:\n```python\n{code}\n```",
                }
            ]
            return []

        # Not all passed - provide feedback
        feedback = build_feedback(results)
        return [{"role": "user", "content": feedback}]

    @vf.stop
    async def task_completed(self, state: vf.State) -> bool:
        """Stop when all training examples pass."""
        return state.get("submitted_answers") is not None
