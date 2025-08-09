# RLVR: Reinforcement Learning with Verifier Rewards

## Project Overview

RLVR is a fresh implementation of multi-step question answering training using the modern [verifiers library](https://github.com/willccbb/verifiers). This project focuses on training language models with reinforcement learning on the MuSiQue dataset for multi-hop reasoning tasks.

## Architecture

### Core Components

1. **Training Infrastructure** (`scripts/train_musique.py`)
   - Modern GRPO training using official verifiers library
   - LoRA fine-tuning support
   - Configurable retrieval strategies
   - WandB integration for experiment tracking

2. **Evaluation System**
   - Custom rubrics for MuSiQue evaluation
   - Exact match and F1 scoring
   - Retrieval quality metrics (recall, precision)
   - Multi-hop difficulty weighting

## Installation & Setup

### Setup Python Environment

```bash
uv venv
source .venv/bin/activate
uv sync
uv pip install flash-attn --no-build-isolation
```

## File Structure

```
├── CLAUDE.md # this file
├── configs # DeepSpeed configuration files
├── docker-compose.yml # docker compose file for the services
├── docs # documentation
├── dvc.yaml # DVC pipeline for the evaluation
├── environments # verifiers environments
├── LICENSE
├── nbs # notebooks
├── outputs # outputs of the training
├── params.yaml # DVC parameters for the evaluation
├── pyproject.toml
├── README.md
├── ruff.toml
├── scripts # training and evaluation scripts
├── services # services for the environment (reranker, wiki search, etc.)
├── src # source code
├── tests
├── tmp
├── uv.lock
```
## Troubleshooting
### Debug Mode

```bash
# Debug with minimal examples
python scripts/train_musique.py --num-train-examples 10 --max-steps 5
```

## Coding Guidelines
===== Philosophy =====
---
description: 
globs: 
alwaysApply: true
---
**My philosophy**

This document outlines my philosophy of software design and implementation. It's not a prescription, nor a claim of superiority. It's a reflection of how I approach building systems with elegance, clarity, and lasting utility. It’s driven by taste—informed by mathematics, tempered by engineering constraints, and matured through hard-earned experience.

---

**1. Abstractions Should Compose**

The most powerful abstractions are those that vanish. They don’t entangle the user in incidental complexity; they lift the problem closer to its essence. Good abstractions compose. You can chain them, combine them, and they retain their integrity. This is true in the UNIX philosophy, where programs are small and pipeable; in functional programming, where functions are first-class citizens; and in algorithm design, where primitives like sorting or hashing can be reused across domains without modification.

The underlying principle is structural invariance—a concept shared with category theory. If a transformation preserves structure (like a functor), it becomes inherently more reusable. Sorting doesn’t care what you sort, as long as the elements are comparable. Matching algorithms don’t care about what the cost matrix represents. They are pure forms. The more your components resemble these, the more broadly they apply.

---

**2. Data First, Behavior Second**

Data is primary. Behavior should be defined as a set of transformations on immutable data. This aligns with the foundational ideas of algebraic data types and functional programming, but also with real-world reasoning: we analyze facts before we apply logic.

In practice, this means preferring  immutable, type-safe containers. They serve as the raw material of computation. Business logic becomes a set of pure functions operating on them. Objects can be useful when they encapsulate protocol-like behavior or manage stateful boundaries, but they should be the exception, not the norm.

---

**3. Prefer Clarity over Cleverness**

Readable code is executable thought. A program is read many more times than it is written, and clarity compounds. This is not a call for verbosity or boilerplate. It’s a call for code that reads close to the domain it operates in, using familiar patterns and structures. If a novice engineer cannot trace the logic without mentally simulating a monad transformer, the abstraction is too clever.

This is also why I prefer comprehensions in Python: they read top-down and clearly express intent. In statically typed languages, method chains offer the same elegance by preserving the container's shape and element type. Expressiveness matters, but never at the cost of mental overhead.

---

**4. Leverage Prior Art Ruthlessly**

Most domain problems have already been solved in another context. Matching, scheduling, searching, routing, indexing—these are well-mapped territories in the landscape of algorithms and data structures. If a problem reduces cleanly to a known abstraction, use it. This isn’t intellectual laziness; it’s compression. It allows us to stand on the shoulders of well-tested, well-understood work.

I derive satisfaction when I recognize a standard algorithm within a domain-specific feature. It means the solution is not arbitrary. It aligns with what Knuth called "programming as a literary form": the elegance comes not from inventing new constructs, but from discovering the most fitting existing ones.

---

**5. Avoid Configuration by Convention**

In large systems, invisible magic becomes technical debt. I prefer explicit wiring, clear boundaries, and visible contracts. Dependency injection should look like data flow. Module boundaries should reflect responsibility, not hierarchy. Convention is useful, but only when it reduces entropy. Beyond that, it breeds fragility.

Composable abstractions resist this fragility. They form a DAG, not a tangle. They can be tested in isolation, reused in orthogonal directions, and combined in surprising ways. The fewer assumptions they make about their environment, the more freely they move.

---

**6. Local Reasoning Is Sacred**

A good design is one where you can understand a component without pulling in the entire system context. Functions should not close over broad scopes. Classes should not mutate global state. Tests should not depend on side-effects.

Category theory calls this referential transparency. Distributed systems call it idempotency. Human cognition calls it sanity. The ability to reason locally is a gift you give your future self and your collaborators.

---

**7. Balance Taste with Practicality**

I admire Haskell, but I write Python. I value purity, but I live in a world of business logic, mutable APIs, and shipping deadlines. The goal is not to simulate a functional language inside an imperative one, but to **borrow ideas** that survive contact with reality.

Taste is not dogma. It’s cultivated instinct. When I choose comprehensions over `map`, or Pydantic over hand-written classes, it’s not because of trend or ideology. It’s because they let me write clearer, safer, more expressive code **in this language**, **in this team**, **under these constraints**.

---

**Final Word**

Code taste is part science, part aesthetics. It can't be fully taught, only sharpened through critique, contrast, and construction. But if I had to condense it:

> Favor abstractions that disappear, data that doesn’t lie, and code that you’re still proud of six months later.

That's the kind of software I want to build. That’s the kind of software I want to maintain. That’s the kind of software that leaves room for both thought and craft.


===== Python =====
---
description: 
globs: *.py
alwaysApply: false
---
Python preferences

- Prefer list/dict comprehension over map/reduce
- When refactoring Python code, try to create modular functions with good abstractions. 
- Prefer `list`, `dict` for type hints over `List`, `Dict`
- Library/module preferences:
    - `uv` for package management
    - `pathlib` for path operations instead of `os` module
    - `pydantic` for data validation
    - `python-dotenv` for environment variables
    - `typer` for CLI
    - `rich` for pretty printing
    - `tqdm` for progress bars
    - `httpx` for HTTP requests
    - `ruff` for linting


===== pytest =====
---
description: 
globs: test*.py
alwaysApply: false
---
Python testing rules

- We use pytest for tests
- Use `@pytest.mark.parametrize` when running same test on different cases

===== uv =====
---
description: Rules for using uv library
globs: 
alwaysApply: false
---
## Scripts

### Declaring script dependencies

The inline metadata format allows the dependencies for a script to be declared in the script itself.

uv supports adding and updating inline script metadata for you. Use `uv add --script` to declare the dependencies for the script:

```
$ uv add --script example.py 'requests<3' 'rich'
```

This will add a `script` section at the top of the script declaring the dependencies using TOML:

example.py

```
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

```
$ uv run example.py
[
│   ('1', 'PEP Purpose and Guidelines'),
...
]
```

Important

When using inline script metadata, even if `uv run` is [used in a _project_](https://docs.astral.sh/uv/concepts/projects/run/), the project's dependencies will be ignored. The `--no-project` flag is not required.

uv also respects Python version requirements:

example.py

```
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

`uv run` will search for and use the required Python version. The Python version will download if it is not installed — see the documentation on [Python versions](https://docs.astral.sh/uv/concepts/python-versions/) for more details.

### Using a shebang to create an executable file

A shebang can be added to make a script executable without using `uv run` — this makes it easy to run scripts that are on your `PATH` or in the current folder.

For example, create a file called `greet` with the following contents

greet

```
#!/usr/bin/env -S uv run --script

print("Hello, world!")
```

Ensure that your script is executable, e.g., with `chmod +x greet`, then run the script:

```
$ ./greet
Hello, world!
```

Declaration of dependencies is also supported in this context, for example:

example

```
#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
import httpx

print(httpx.get("https://example.com"))
```

## References

- [Verifiers Library](https://github.com/willccbb/verifiers) - Official documentation