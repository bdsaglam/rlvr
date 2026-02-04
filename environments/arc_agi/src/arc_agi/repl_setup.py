"""REPL setup code for ARC-AGI environments.

This module contains the Python code that gets injected into the REPL
to set up task data, utility functions, and submission handling.
"""

# Setup code for local (in-process) REPL execution
LOCAL_REPL_SETUP_CODE = '''\
import numpy as np
import json

# --- ARC task data (auto-injected) ---
train_pairs = {train_pairs_json}
train_inputs = [np.array(p["input"]) for p in train_pairs]
train_outputs = [np.array(p["output"]) for p in train_pairs]
test_inputs = [np.array(t) for t in {test_inputs_json}]
num_test_cases = {num_test_cases}

# --- Submission storage ---
_submitted_answers = None

# --- Utility functions ---
COLOR_NAMES = {{
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "grey", 6: "pink", 7: "orange", 8: "cyan", 9: "maroon",
}}

def show(grid, title=None):
    """Pretty-print a grid (numpy array or list of lists)."""
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if title:
        print("--- " + str(title) + " ---")
    for row in grid:
        print(" ".join(str(c) for c in row))

def show_pair(input_grid, output_grid, title=None):
    """Print an input-output pair side by side (or stacked if too wide)."""
    inp = np.array(input_grid)
    out = np.array(output_grid)

    if title:
        print("=== " + str(title) + " ===")

    inp_lines = [" ".join(str(int(c)) for c in row) for row in inp]
    out_lines = [" ".join(str(int(c)) for c in row) for row in out]

    inp_width = max(len(line) for line in inp_lines) if inp_lines else 0
    out_width = max(len(line) for line in out_lines) if out_lines else 0

    if inp_width + out_width < 60:
        max_height = max(len(inp_lines), len(out_lines))
        while len(inp_lines) < max_height:
            inp_lines.append(" " * inp_width)
        while len(out_lines) < max_height:
            out_lines.append(" " * out_width)

        print("Input:" + " " * (inp_width - 5) + "  -->  Output:")
        for i_line, o_line in zip(inp_lines, out_lines):
            print(i_line.ljust(inp_width) + "  -->  " + o_line)
    else:
        print("Input:")
        for line in inp_lines:
            print(line)
        print()
        print("Output:")
        for line in out_lines:
            print(line)
    print()

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

def _grid_diff(pred, expected, max_display=15):
    """Generate a visual diff showing mismatches."""
    rows, cols = pred.shape
    lines = []
    show_rows = min(rows, max_display)
    show_cols = min(cols, max_display)

    for i in range(show_rows):
        row_parts = []
        for j in range(show_cols):
            if pred[i, j] == expected[i, j]:
                row_parts.append(str(int(pred[i, j])).rjust(2))
            else:
                row_parts.append(str(int(pred[i, j])) + "/" + str(int(expected[i, j])))
        lines.append(" ".join(row_parts))

    if rows > max_display or cols > max_display:
        lines.append("... (truncated, full shape: " + str(rows) + "x" + str(cols) + ")")

    return "\\n".join(lines)

def verify(transform_func):
    """Verify a transform function against ALL training examples.

    Call this to test your transform function before submitting.
    Returns a detailed report showing pass/fail status for each example.

    Example:
        def transform(grid):
            return np.rot90(grid)
        verify(transform)
    """
    results = []
    total_passed = 0
    total_examples = len(train_pairs)

    for i, pair in enumerate(train_pairs):
        example_num = i + 1
        input_grid = np.array(pair["input"])
        expected_output = np.array(pair["output"])

        try:
            predicted = transform_func(input_grid.copy())
            predicted = np.array(predicted)

            if predicted.shape != expected_output.shape:
                results.append(
                    "Example " + str(example_num) + ": FAIL (shape mismatch)\\n" +
                    "  Input shape:    " + str(input_grid.shape) + "\\n" +
                    "  Expected shape: " + str(expected_output.shape) + "\\n" +
                    "  Got shape:      " + str(predicted.shape)
                )
            elif np.array_equal(predicted, expected_output):
                total_passed += 1
                results.append("Example " + str(example_num) + ": PASS")
            else:
                accuracy = np.mean(predicted == expected_output)
                num_wrong = int(np.sum(predicted != expected_output))
                total_cells = predicted.size
                diff_viz = _grid_diff(predicted, expected_output)

                results.append(
                    "Example " + str(example_num) + ": FAIL\\n" +
                    "  Accuracy: " + str(round(accuracy * 100, 1)) + "% (" +
                    str(total_cells - num_wrong) + "/" + str(total_cells) + " cells correct)\\n" +
                    "  Diff (predicted/expected for mismatches):\\n" + diff_viz
                )
        except Exception as e:
            results.append(
                "Example " + str(example_num) + ": ERROR\\n" +
                "  Exception: " + type(e).__name__ + ": " + str(e)
            )

    status = "ALL PASSED" if total_passed == total_examples else "FAILED (" + str(total_passed) + "/" + str(total_examples) + " passed)"
    summary = "=== Verification: " + status + " ===\\n"

    print(summary + "\\n" + "\\n\\n".join(results))

def check_example(transform_func, example_index):
    """Check transform on a single training example (0-indexed).

    Useful for debugging a specific failing example.

    Example:
        check_example(transform, 0)  # Check first training example
    """
    if example_index < 0 or example_index >= len(train_pairs):
        print("Invalid example index. Valid range: 0 to " + str(len(train_pairs) - 1))
        return

    pair = train_pairs[example_index]
    input_arr = np.array(pair["input"])
    expected_arr = np.array(pair["output"])

    print("=== Example " + str(example_index + 1) + " ===")
    print("Input shape:", input_arr.shape)
    print("Expected output shape:", expected_arr.shape)
    print()

    try:
        predicted = transform_func(input_arr.copy())
        predicted = np.array(predicted)

        if predicted.shape != expected_arr.shape:
            print("FAIL: Shape mismatch")
            print("  Expected:", expected_arr.shape)
            print("  Got:     ", predicted.shape)
            return

        if np.array_equal(predicted, expected_arr):
            print("PASS: Output matches expected exactly!")
            return

        accuracy = np.mean(predicted == expected_arr)
        diff_viz = _grid_diff(predicted, expected_arr)

        print("FAIL: Output does not match")
        print("  Accuracy:", str(round(accuracy * 100, 1)) + "%")
        print("  Diff (predicted/expected for mismatches):")
        print(diff_viz)
    except Exception as e:
        print("ERROR:", type(e).__name__ + ":", str(e))

def submit_answer(answers):
    """Submit final answer grids for both training and test examples.

    Args:
        answers: Dict with "train" and "test" keys, each containing a list
                 of output grids. Each grid can be a numpy array or list of lists.
                 Values must be integers 0-9.

    Example:
        def transform(grid):
            return np.rot90(grid)
        train_preds = [transform(inp) for inp in train_inputs]
        test_preds = [transform(inp) for inp in test_inputs]
        submit_answer({{"train": train_preds, "test": test_preds}})
    """
    global _submitted_answers
    def convert_grids(grid_list):
        result = []
        for grid in grid_list:
            if hasattr(grid, "tolist"):
                grid = grid.tolist()
            result.append([[int(c) for c in row] for row in grid])
        return result

    _submitted_answers = {{
        "train": convert_grids(answers.get("train", [])),
        "test": convert_grids(answers.get("test", [])),
    }}
    print("Answers submitted.")
'''
