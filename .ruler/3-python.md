## Python

- When refactoring Python code, try to create modular functions with good abstractions. 
- Ensure virtual environment is activated before running any Python code. Either run `source .venv/bin/activate` in the root of the repository or use `uv run` when running scripts.
- When creating singleton objects, use `@cache` decorator:
```python
from functools import cache

@cache
def get_singleton_instance():
    return MySingleton()
```

- Prefer list/dict comprehension over map/reduce
- Prefer `pathlib` for path operations instead of `os` module
- Prefer `list`, `dict` for type hints over `List`, `Dict`
- Prefer `T | None` over `Optional[T]`
- Library preferences:
    - `uv` for environment management and package management
    - `pytest` for testing
    - `typer` for CLI scripts
    - `rich` for pretty printing
    - `httpx` for HTTP requests
    - `pydantic` for data validation
    - `uuid6` for unique IDs. use `uuid7()` function in it.
    - `python-dotenv` for loading environment variables from `.env` file
    - `ruff` for linting and formatting
    - `pre-commit` for pre-commit hooks

### pytest

- We use pytest for tests
- Use `@pytest.mark.parametrize` when running same test on different cases
- Use function tests like `def test_happy_path_something()` rather than class-based tests

### uv

#### Declaring script dependencies

The inline metadata format allows the dependencies for a script to be declared in the script itself.

uv supports adding and updating inline script metadata for you. Use `uv add --script` to declare the dependencies for the script:

```sh
uv add --script example.py 'requests<3' 'rich'
```

This will add a `script` section at the top of the script declaring the dependencies using TOML:

example.py

```python
# /// script
# dependencies = [
#   "requests<3",
#   "rich",
# ]
# ///

import requests
from rich.pretty import pprint

resp = requests.get("https://peps.python.org/api/peps.json")
data = resp.json()
pprint([(k, v["title"]) for k, v in data.items()][:10])
```

uv will automatically create an environment with the dependencies necessary to run the script, e.g.:

```sh
uv run example.py
[
│   ('1', 'PEP Purpose and Guidelines'),
...
]
```

Important

When using inline script metadata, even if `uv run` is [used in a _project_](mdc:https:/docs.astral.sh/uv/concepts/projects/run), the project's dependencies will be ignored. The `--no-project` flag is not required.

uv also respects Python version requirements:

example.py

```python
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

# Use some syntax added in Python 3.12
type Point = tuple[float, float]
print(Point)
```

Note

The `dependencies` field must be provided even if empty.

`uv run` will search for and use the required Python version. The Python version will download if it is not installed — see the documentation on [Python versions](mdc:https:/docs.astral.sh/uv/concepts/python-versions) for more details.

#### Using a shebang to create an executable file

A shebang can be added to make a script executable without using `uv run` — this makes it easy to run scripts that are on your `PATH` or in the current folder.

For example, create a file called `greet` with the following contents

greet

```py
#!/usr/bin/env -S uv run --script

print("Hello, world!")
```

Ensure that your script is executable, e.g., with `chmod +x greet`, then run the script:

```sh
./greet
Hello, world!
```

Declaration of dependencies is also supported in this context, for example:

example

```py
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
import httpx

print(httpx.get("https://example.com"))
```