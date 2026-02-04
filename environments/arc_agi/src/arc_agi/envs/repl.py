"""REPL-based environment for ARC-AGI.

Provides a persistent Python REPL with pre-loaded task data and utilities.
The model uses a `python` tool to execute code and `submit_answer()` to submit.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import traceback
from typing import Any

import verifiers as vf

from ..repl_setup import LOCAL_REPL_SETUP_CODE

# ---------------------------------------------------------------------------
# Local Python REPL
# ---------------------------------------------------------------------------


class LocalPythonREPL:
    """In-process persistent Python REPL for local execution."""

    def __init__(self):
        self.namespace: dict[str, Any] = {"__name__": "__main__"}
        self.execution_count = 0

    def execute(self, code: str) -> str:
        """Execute code and return formatted output."""
        self.execution_count += 1
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        result_value = None
        status = "ok"

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                module_ast = ast.parse(code, mode="exec")
                body = list(module_ast.body)
                trailing_expr = None
                if body and isinstance(body[-1], ast.Expr):
                    trailing_expr = body.pop()
                if body:
                    exec_module = ast.Module(body=body, type_ignores=[])
                    exec(compile(exec_module, "<cell>", "exec"), self.namespace, self.namespace)
                if trailing_expr is not None:
                    value = eval(
                        compile(ast.Expression(trailing_expr.value), "<cell>", "eval"),
                        self.namespace,
                        self.namespace,
                    )
                    if value is not None:
                        result_value = repr(value)
        except Exception:
            status = "error"
            result_value = traceback.format_exc()

        # Format output
        parts: list[str] = []
        stdout = stdout_buffer.getvalue().rstrip()
        if stdout:
            parts.append(stdout)
        stderr = stderr_buffer.getvalue().rstrip()
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        if status == "error" and result_value:
            parts.append(result_value.rstrip())
        elif status == "ok" and result_value is not None:
            parts.append(f"Out[{self.execution_count}]: {result_value}")
        if not parts:
            parts.append("(no output)")
        return "\n".join(parts)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are solving an ARC-AGI puzzle. You are given training examples (input/output
grid pairs) that demonstrate a transformation pattern. Your task is to discover
the pattern and apply it to the test input(s) to produce the correct output grid(s).

Grids are 2D arrays of integers 0-9 representing colors:
  0=black 1=blue 2=red 3=green 4=yellow 5=grey 6=pink 7=orange 8=cyan 9=maroon

Grids are shown as space-separated digits inside <Diagram> tags.

You have ONE tool available: `python` — a persistent Python REPL.

The REPL has these pre-loaded variables:
- `train_pairs`: list of {{"input": grid, "output": grid}} dicts
- `train_inputs`: list of numpy arrays (one per training input)
- `train_outputs`: list of numpy arrays (one per training output)
- `test_inputs`: list of numpy arrays (one per test input)
- `np` (numpy) is imported

The REPL also has these utility functions available (no need to import them):
- `show(grid, title=None)`: pretty-print a grid
- `show_pair(input_grid, output_grid, title=None)`: side-by-side input/output display
- `grid_shape(grid)`: return (rows, cols)
- `verify(transform_func)`: test your transform against ALL training examples
- `check_example(transform_func, index)`: debug a specific training example (0-indexed)

Submission:
- `submit_answer({{"train": [...], "test": [...]}})`: submit final predictions

Strategy:
1. Study the training examples carefully. Look for patterns in how inputs transform
   to outputs: spatial transforms, color mappings, object detection, symmetry,
   counting, filling, etc.
2. Write a `transform(grid)` function that implements your hypothesis.
3. VERIFY your solution with `verify(transform)` — it must pass ALL training examples.
4. If verification fails, debug with `check_example(transform, i)` and refine.
5. Only after verification passes, submit your answer.

Example:
```python
# 1. Implement transform
def transform(grid):
    return np.rot90(grid)

# 2. Verify against training examples
verify(transform)  # Must show "ALL PASSED" before submitting!

# 3. If verification passes, submit
train_preds = [transform(inp) for inp in train_inputs]
test_preds = [transform(inp) for inp in test_inputs]
submit_answer({{"train": train_preds, "test": test_preds}})
```

Rules:
- The ONLY tool is `python`. There is no separate submit_answer tool.
- ALWAYS verify your transform passes ALL training examples before submitting.
- Call `submit_answer()` exactly once inside your Python code.
- Submit a dict with "train" and "test" keys, each containing a list of grids.
- Each answer grid must contain integers 0-9."""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ArcAgiREPLEnv(vf.StatefulToolEnv):
    """Local REPL-based environment for ARC-AGI (no Docker required)."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, **kwargs):
        super().__init__(system_prompt=system_prompt, **kwargs)
        self.add_tool(self.python, args_to_skip=["state"])

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        state["submitted_answers"] = None

        # Create per-rollout REPL
        repl = LocalPythonREPL()
        state["_repl"] = repl

        # Run setup code
        info = state["info"]
        setup_code = LOCAL_REPL_SETUP_CODE.format(
            train_pairs_json=json.dumps(info["train_pairs"]),
            test_inputs_json=json.dumps(info["test_inputs"]),
            num_test_cases=info["num_test_cases"],
        )
        repl.execute(setup_code)
        return state

    async def python(self, code: str, state: vf.State) -> str:
        """Execute Python code in the local REPL."""
        repl: LocalPythonREPL = state["_repl"]
        result = repl.execute(code)

        # Check if submit_answer was called
        submitted = repl.namespace.get("_submitted_answers")
        if submitted is not None and state.get("submitted_answers") is None:
            state["submitted_answers"] = submitted
            state["final_env_response"] = [
                {"role": "tool", "content": f"Submitted answers:\n{json.dumps(submitted, indent=2)}"}
            ]

        return result

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Inject state into python tool calls."""
        if tool_name == "python":
            tool_args["state"] = state
        return tool_args

    @vf.stop
    async def answer_submitted(self, state: vf.State) -> bool:
        """Stop when submit_answer is called."""
        return state.get("submitted_answers") is not None

