"""Tests for the ARC-AGI REPL environment."""

import pytest

from arc_agi.envs.repl import ARC_SANDBOX_CODE, parse_response
from arc_agi.subprocess_interpreter import SubprocessInterpreter


class TestParseResponse:
    """Tests for response parsing."""

    def test_parse_structured_response(self):
        """Test parsing a well-formed structured response."""
        text = """
[[ ## reasoning ## ]]
This is my reasoning about the problem.
I think the pattern is rotation.

[[ ## code ## ]]
```python
print("hello")
x = 1 + 2
```
"""
        sections = parse_response(text)
        assert "rotation" in sections["reasoning"]
        assert 'print("hello")' in sections["code"]
        assert "x = 1 + 2" in sections["code"]

    def test_parse_code_without_fences(self):
        """Test parsing code section without markdown fences."""
        text = """
[[ ## reasoning ## ]]
Simple test.

[[ ## code ## ]]
x = 42
print(x)
"""
        sections = parse_response(text)
        assert "x = 42" in sections["code"]

    def test_fallback_to_code_fences(self):
        """Test fallback extraction when no section markers present."""
        text = """
Here's some code:
```python
result = 1 + 1
```
"""
        sections = parse_response(text)
        assert "result = 1 + 1" in sections["code"]

    def test_empty_response(self):
        """Test handling of empty response."""
        sections = parse_response("")
        assert sections["reasoning"] == ""
        assert sections["code"] == ""


class TestSandboxCodeInjection:
    """Tests for task data injection into sandbox."""

    def test_task_variable_injection(self):
        """Test that task data is correctly injected and accessible."""
        import json

        # Sample task data
        task_data = {
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
                {"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]},
            ],
            "test": [
                {"input": [[4, 5], [5, 4]]},
            ],
        }

        # Format sandbox code
        sandbox_code = ARC_SANDBOX_CODE.format(task_json=json.dumps(task_data))

        # Create interpreter and execute
        interpreter = SubprocessInterpreter(sandbox_code=sandbox_code)
        try:
            interpreter.start()

            # Test that task variable exists and has correct structure
            result = interpreter.execute("len(task['train'])")
            assert "2" in result

            result = interpreter.execute("len(task['test'])")
            assert "1" in result

            # Test train data structure
            result = interpreter.execute("task['train'][0]['input']")
            assert "[[0, 1], [1, 0]]" in result

            result = interpreter.execute("task['train'][0]['output']")
            assert "[[1, 0], [0, 1]]" in result

            # Test test data structure (no output key)
            result = interpreter.execute("task['test'][0]['input']")
            assert "[[4, 5], [5, 4]]" in result

            result = interpreter.execute("'output' in task['test'][0]")
            assert "False" in result

        finally:
            interpreter.shutdown()

    def test_helper_functions_available(self):
        """Test that helper functions are available in sandbox."""
        import json

        task_data = {
            "train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}],
            "test": [{"input": [[5, 6], [7, 8]]}],
        }

        sandbox_code = ARC_SANDBOX_CODE.format(task_json=json.dumps(task_data))
        interpreter = SubprocessInterpreter(sandbox_code=sandbox_code)

        try:
            interpreter.start()

            # Test format_grid
            result = interpreter.execute("format_grid([[1, 2], [3, 4]])")
            assert "1,2" in result
            assert "3,4" in result

            # Test accuracy
            result = interpreter.execute("accuracy([[1, 2], [3, 4]], [[1, 2], [3, 4]])")
            assert "1.0" in result

            result = interpreter.execute("accuracy([[1, 2], [3, 4]], [[0, 0], [0, 0]])")
            assert "0.0" in result

            # Test soft_accuracy
            result = interpreter.execute("soft_accuracy([[1, 2], [3, 4]], [[1, 2], [0, 0]])")
            assert "0.5" in result

            # Test numpy is available
            result = interpreter.execute("np.array([[1, 2], [3, 4]]).shape")
            assert "(2, 2)" in result

        finally:
            interpreter.shutdown()

    def test_grid_manipulation(self):
        """Test typical grid manipulation workflow."""
        import json

        task_data = {
            "train": [{"input": [[0, 0], [0, 0]], "output": [[1, 1], [1, 1]]}],
            "test": [{"input": [[0, 0, 0], [0, 0, 0]]}],
        }

        sandbox_code = ARC_SANDBOX_CODE.format(task_json=json.dumps(task_data))
        interpreter = SubprocessInterpreter(sandbox_code=sandbox_code)

        try:
            interpreter.start()

            # Simulate typical workflow
            interpreter.execute("pred = np.array(task['test'][0]['input']).copy()")
            interpreter.execute("pred[:, :] = 1")
            result = interpreter.execute("pred.tolist()")
            assert "[[1, 1, 1], [1, 1, 1]]" in result

            # Verify against training
            interpreter.execute("train_pred = np.array(task['train'][0]['input']).copy()")
            interpreter.execute("train_pred[:, :] = 1")
            result = interpreter.execute("accuracy(train_pred, task['train'][0]['output'])")
            assert "1.0" in result

        finally:
            interpreter.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
