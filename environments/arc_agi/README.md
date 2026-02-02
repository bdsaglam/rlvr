# ARC-AGI Environment

Trains LLMs to solve [ARC-AGI](https://arcprize.org/) visual pattern reasoning puzzles using reinforcement learning. The model gets a persistent Python REPL with pre-loaded task data and utilities.

## Usage

```bash
prime eval run arc-agi --data_dir data/arc-dummy --split training --max_turns 10
```

## How It Works

The model interacts via a single `python` tool (persistent REPL). The REPL comes pre-loaded with:

- `train_pairs`, `train_inputs`, `train_outputs` — training examples as numpy arrays
- `test_inputs` — test inputs to solve
- `show(grid)`, `grid_shape(grid)`, `unique_colors(grid)` — utility functions
- `submit_answer(answers)` — call with a list of grids to submit

The model can compute answers as variables and submit directly:

```python
result = np.rot90(test_inputs[0])
submit_answer([result])
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `data_dir` | `data/arc-prize-2025` | Path to ARC data directory |
| `split` | `training` | Data split (`training` or `evaluation`) |
| `eval_data_dir` | `None` | Separate data dir for evaluation |
| `eval_split` | `evaluation` | Evaluation data split |
| `grid_format` | `json` | Grid display format: `json` or `ascii` |
| `reward_mode` | `binary` | Reward weighting: `binary`, `partial`, or `combined` |
| `max_turns` | `10` | Maximum interaction turns |

## Reward Modes

- **binary**: 1.0 only if all test outputs match exactly
- **partial**: Cell-level accuracy averaged across test cases
- **combined**: 50/50 mix of exact match and cell accuracy

## Data Format

Expects ARC-AGI JSON files in the data directory:
- `arc-agi_{split}_challenges.json`
- `arc-agi_{split}_solutions.json`
