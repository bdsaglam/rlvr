"""Sandbox for executing Python code in a subprocess.

Provides isolated code execution with timeout support.
Based on epiq's neuro_symbolic_sandbox but simplified for RL training
(no DSPy/LLM sub-calls needed during evaluation).
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

Grid = list[list[int]]


def run(
    code: str,
    input_grid: Grid,
    timeout_s: float = 2.0,
) -> tuple[bool, str]:
    """Run transform code in a subprocess.

    Args:
        code: Python code containing a transform(grid: np.ndarray) -> np.ndarray function.
        input_grid: Input grid as list of lists.
        timeout_s: Timeout in seconds.

    Returns:
        Tuple of (success: bool, result_or_error: str).
        If success, result is JSON-encoded output grid.
        If failure, result is error message.
    """
    script = _build_script(code)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "transform.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)

        env = {
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
        }

        try:
            result = subprocess.run(
                [sys.executable, path],
                input=json.dumps({"input": input_grid}).encode(),
                capture_output=True,
                cwd=td,
                env=env,
                timeout=timeout_s,
            )
            stdout = result.stdout
            stderr = result.stderr
        except subprocess.TimeoutExpired:
            return False, "Timeout: code execution exceeded time limit"

        if result.returncode != 0:
            error_msg = (stderr.decode() or stdout.decode()).strip()
            if len(error_msg) > 1000:
                error_msg = error_msg[:1000] + "... [truncated]"
            return False, error_msg

        stdout_str = stdout.decode()

        try:
            payload = json.loads(stdout_str)
            ok = bool(payload.get("ok"))

            if not ok and payload.get("error"):
                return False, payload.get("error")

            result_str = json.dumps(payload.get("result"))
            return ok, result_str
        except Exception as e:
            stderr_str = stderr.decode()
            error_details = f"Invalid output: {e}\nstdout: {stdout_str[:500]}\nstderr: {stderr_str[:500]}"
            return False, error_details


def _build_script(code: str) -> str:
    """Build the complete Python script with runtime and user code."""
    return f'''
# Sandbox runtime for transform execution

import json
import sys
import numpy as np
import scipy

# ============ User Code ============
{code}

# ============ Main Execution ============
if __name__ == '__main__':
    def _output_json(ok, result=None, error=None):
        """Output result in JSON format."""
        output = {{
            "ok": ok,
            "result": result,
            "error": error,
        }}
        print(json.dumps(output))

    try:
        data = json.load(sys.stdin)
        input_grid = np.array(data['input'])
        result = transform(input_grid)

        # Ensure result is a numpy array and convert to list
        if hasattr(result, 'tolist'):
            result_list = result.tolist()
        else:
            result_list = list(result)

        # Ensure all values are ints
        result_list = [[int(c) for c in row] for row in result_list]

        _output_json(True, result=result_list)

    except Exception as e:
        import traceback
        _output_json(False, error=f"{{type(e).__name__}}: {{e}}")
'''


def execute_transform(code: str, input_grid: Grid, timeout_s: float = 2.0) -> tuple[Grid | None, str | None]:
    """Execute transform function on an input grid.

    Returns:
        (output_grid, error_message) - one will be None
    """
    ok, result = run(code, input_grid, timeout_s=timeout_s)

    if not ok:
        return None, result

    try:
        output = json.loads(result)
        return output, None
    except Exception as e:
        return None, f"Failed to parse output: {e}"


def evaluate_on_train(code: str, train_pairs: list[dict], timeout_s: float = 2.0) -> list[dict]:
    """Evaluate transform function on all training examples.

    Returns list of results, each with:
        - index: example index (0-based)
        - passed: bool
        - expected: Grid
        - predicted: Grid | None
        - error: str | None
        - accuracy: float (cell-level, 0.0 if shape mismatch or error)
    """
    results = []
    for i, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        expected = pair["output"]

        predicted, error = execute_transform(code, input_grid, timeout_s=timeout_s)

        if error:
            results.append({
                "index": i,
                "passed": False,
                "expected": expected,
                "predicted": None,
                "error": error,
                "accuracy": 0.0,
            })
            continue

        # Compare shapes
        pred_arr = np.array(predicted)
        exp_arr = np.array(expected)

        if pred_arr.shape != exp_arr.shape:
            results.append({
                "index": i,
                "passed": False,
                "expected": expected,
                "predicted": predicted,
                "error": f"Shape mismatch: got {pred_arr.shape}, expected {exp_arr.shape}",
                "accuracy": 0.0,
            })
            continue

        # Calculate accuracy
        matches = pred_arr == exp_arr
        accuracy = float(np.mean(matches))
        passed = bool(np.all(matches))

        results.append({
            "index": i,
            "passed": passed,
            "expected": expected,
            "predicted": predicted,
            "error": None,
            "accuracy": accuracy,
        })

    return results
