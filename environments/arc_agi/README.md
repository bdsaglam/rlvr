# ARC-AGI Environment

Trains LLMs to solve [ARC-AGI](https://arcprize.org/) visual pattern reasoning puzzles using reinforcement learning.

## Environment Types

### Iterative (`env_type="iterative"`, default)

The LLM writes a `transform` function in markdown code blocks. The function is automatically evaluated on training examples:
- If all pass → task complete, function applied to test inputs
- If not → feedback provided (pass/fail per example, diff visualization)

```bash
prime eval run arc-agi -x '{"dataset":"arc-dummy"}' -n 1 -r 1
```

### REPL (`env_type="repl"`)

The LLM uses a `python` tool to interact with a persistent REPL pre-loaded with task data and utilities. Must manually verify and submit.

```bash
prime eval run arc-agi -x '{"dataset":"arc-dummy","env_type":"repl"}' -n 1 -r 1
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | `arc-prize-2025` | ARC data folder name under `environments/arc_agi/data` |
| `split` | `training` | Data split (`training` or `evaluation`) |
| `eval_dataset` | `None` | Separate ARC data folder name for evaluation |
| `eval_split` | `evaluation` | Evaluation data split |
| `reward_mode` | `binary` | `binary`, `partial`, or `combined` |
| `max_turns` | `10` | Maximum interaction turns |
| `env_type` | `iterative` | `iterative` or `repl` |
| `timeout_s` | `2.0` | Code execution timeout (iterative only) |


## Reward Modes

- **binary**: 1.0 only if all test outputs match exactly
- **partial**: Cell-level accuracy averaged across test cases
- **combined**: 50/50 mix of exact match and cell accuracy

## Data Format

Expects ARC-AGI JSON files in each folder under `environments/arc_agi/data`:
- `arc-agi_{split}_challenges.json`
- `arc-agi_{split}_solutions.json`

## Module Structure

```
arc_agi/
├── __init__.py      # Package entry point
├── env.py           # load_environment() dispatcher
├── data.py          # Data loading and formatting
├── rewards.py       # Reward functions and rubric
├── sandbox.py       # Subprocess code execution
├── repl_setup.py    # REPL setup code templates
└── envs/
    ├── iterative.py # Iterative refinement environment
    └── repl.py      # REPL-based environment
```
