# Training & Evaluation with Prime-RL

This guide covers how to train models with reinforcement learning using [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) and evaluate them with [verifiers](https://github.com/PrimeIntellect-ai/verifiers).

## Architecture Overview

Prime-RL runs three processes that communicate asynchronously:

1. **Inference** — vLLM server generating rollouts on GPU(s)
2. **Orchestrator** — manages rollout generation, reward computation, and batching
3. **Trainer** — updates model weights via RL on GPU(s)

The trainer broadcasts updated weights to the inference server after each step. The orchestrator coordinates the data flow between them.

## Config File Structure

Training is configured via a single TOML file. Key sections:

```toml
# GPU assignment
inference_gpu_ids = [0, 1, 2]
trainer_gpu_ids = [3]

# Global settings
max_steps = 500
seq_len = 16384

# W&B logging
[wandb]
project = "my-project"
name = "my-run"

# Base model (used by all components)
[model]
name = "Qwen/Qwen2.5-7B-Instruct"

# Trainer settings
[trainer.optim]
lr = 1e-5
weight_decay = 0.0

[trainer.tokenizer]
name = "Qwen/Qwen2.5-7B-Instruct"

# Activation checkpointing (reduces memory)
[trainer.model.ac]
freq = 1  # every layer = full AC

# LoRA (optional)
[trainer.model.lora]
rank = 32
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Orchestrator settings
[orchestrator]
batch_size = 32               # total samples per training step
rollouts_per_example = 16     # rollouts per dataset example (must divide batch_size)
oversampling_factor = 2.0     # oversample to ensure enough valid rollouts

[orchestrator.sampling]
max_tokens = 1024
temperature = 0.5

[orchestrator.buffer]
online_difficulty_filtering = true

# Environment(s)
[[orchestrator.env]]
id = "my-env"
args = { dataset_name = "train", difficulty = "hard" }

# Inference server (vLLM)
[inference]
gpu_memory_utilization = 0.6

[inference.model]
name = "Qwen/Qwen2.5-7B-Instruct"
enable_auto_tool_choice = true
tool_call_parser = "hermes"
max_model_len = 16384
```

## Running Training

```bash
uv run rl @ configs/prime-rl/my-config.toml
```

This starts all three processes (inference, orchestrator, trainer) and shows trainer logs.

### Constraints

- `batch_size` must be divisible by `rollouts_per_example`
- `seq_len` must match or exceed `max_model_len`
- All three model name fields (top-level, trainer tokenizer, inference) should generally match

## Key Parameters

### Orchestrator

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `batch_size` | Samples per training step | Must be divisible by `rollouts_per_example` |
| `rollouts_per_example` | Completions per input example | Higher = better advantage estimates |
| `oversampling_factor` | Multiplier for rollout generation | Compensates for filtered/failed rollouts |
| `trajectory_strategy` | `"interleaved"` (default) or `"branching"` | Use `"branching"` for thinking models (Qwen3, etc.) |

### Trainer

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `lr` | Learning rate | Start with 1e-6 for stability, up to 1e-5 |
| `weight_decay` | Weight decay | 0.0 is common for RL |
| `optimization_dtype` | Dtype for optimization | `"bfloat16"` or `"float32"` (default) |
| `reduce_dtype` | Dtype for gradient reduction | `"bfloat16"` or `"float32"` (default) |
| `ac.freq` | Activation checkpointing frequency | `1` = full AC (every layer) |
| `lora.rank` | LoRA rank | 16-64 typical |
| `lora.alpha` | LoRA alpha | Often same as rank or 2x rank |

### Inference

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `dtype` | Model dtype for vLLM | `"auto"` (default), `"float16"`, `"bfloat16"`, `"float32"` |
| `gpu_memory_utilization` | Fraction of GPU memory for KV cache | 0.6-0.9 typical |
| `max_model_len` | Max context length | Must be ≤ model's max and ≥ `seq_len` |

### Sampling

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `max_tokens` | Max tokens per model turn | Not total sequence — per turn |
| `temperature` | Sampling temperature | 0.5-1.0; higher = more diverse rollouts |

## Thinking Models (Qwen3, DeepSeek-R1, etc.)

When using models that produce `<think>...</think>` blocks:

1. Set `trajectory_strategy = "branching"` in orchestrator config. Qwen3's chat template strips thinking tokens from prior turns, which breaks the interleaved strategy's exact-prefix invariant.

2. Add `reasoning_parser` to inference model config:
   ```toml
   [inference.model]
   reasoning_parser = "qwen3"  # or "deepseek_r1"
   ```

3. Tool call parser for Qwen3 is still `"hermes"`.

## Activation Checkpointing

Reduces GPU memory by recomputing activations during backward pass instead of storing them. Add to config:

```toml
[trainer.model.ac]
freq = 1  # checkpoint every layer (full AC)
```

`freq = 2` would checkpoint every other layer (less memory savings, less compute overhead).

## Data Types (dtype)

Trainer and inference have separate dtype settings.

### Trainer dtype

Set under `[trainer.model]`:

```toml
[trainer.model]
optimization_dtype = "bfloat16"  # or "float32" (default)
reduce_dtype = "bfloat16"        # or "float32" (default)
```

- `optimization_dtype` — precision for forward/backward passes
- `reduce_dtype` — precision for gradient reduction across GPUs

Only `"bfloat16"` and `"float32"` are supported (no `float16`).

Source: [`prime_rl/trainer/config.py:219-231`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/trainer/config.py#L219-L231)

### Inference dtype

Set under `[inference.model]`:

```toml
[inference.model]
dtype = "float16"  # or "auto" (default), "bfloat16", "float32"
```

This is passed to vLLM's `--dtype` flag. `"auto"` uses FP16 for FP32/FP16 models, BF16 for BF16 models.

Source: [`prime_rl/inference/config.py:52-57`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/inference/config.py#L52-L57)

## Checkpointing & Resuming

```toml
[trainer.ckpt]
interval = 10        # save every N steps
keep_last = 3        # keep last N checkpoints
keep_interval = 50   # keep every Nth step permanently
```

Resume from a checkpoint:
```bash
uv run rl @ configs/prime-rl/my-config.toml --trainer.ckpt.resume-step 50
```

Checkpoints are saved to `outputs/weights/step_{N}/` and `outputs/checkpoints/step_{N}/`.

## Evaluation

### With `prime eval`

You must start a vLLM server with the model before running eval. Use `vllm serve` directly (`vf-vllm` may have import compatibility issues with newer vLLM versions):

```bash
# Start vLLM server first (in a separate terminal)
CUDA_VISIBLE_DEVICES=0,1,2 vllm serve Qwen/Qwen2.5-7B-Instruct \
    --port 8007 \
    --data-parallel-size 3 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```

Then run evaluation against it using `-b` to point at the local server:

```bash
# Against a local vLLM server
prime eval run vf-musique -m Qwen/Qwen2.5-7B-Instruct -b http://0.0.0.0:8007/v1

# Against a hosted/API model (uses Prime Inference by default)
prime eval run vf-musique -m gpt-5-nano
```

For thinking models (Qwen3, etc.), add the reasoning parser:
```bash
vllm serve mit-oasys/rlm-qwen3-8b-v0.1 \
    --port 8007 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --enforce-eager
```

### Evaluating Training Checkpoints

Start the inference server with the base model, then evaluate each checkpoint:

```bash
# Terminal 1: start vLLM (see above)

# Terminal 2: evaluate all checkpoints
uv run eval \
  --client.base-url http://localhost:8007/v1 \
  --weights-dir outputs/weights \
  --env.id my-env
```

To evaluate specific steps:
```bash
uv run eval \
  --weights-dir outputs/weights \
  --steps 10,50,100 \
  --no-eval-base
```

### Online Evaluation During Training

Add eval config to the orchestrator:

```toml
[orchestrator.eval]
interval = 100
rollouts_per_example = 1
num_examples = 100

[[orchestrator.eval.env]]
id = "my-env"
args = { split = "validation" }
```

### Viewing Results

```bash
prime eval tui  # terminal UI for browsing results
```

## Environment Setup

Environments are Python packages installed into the project. Each must expose a `load_environment()` function:

```python
import verifiers as vf

def load_environment(difficulty: str = "easy", **kwargs) -> vf.Environment:
    dataset = prepare_dataset(difficulty)
    rubric = vf.Rubric(funcs=[my_reward_func])
    return vf.ToolEnv(dataset=dataset, tools=[my_tool], rubric=rubric, max_turns=10)
```

Install local environments:
```bash
prime env install my-env  # from ./environments/my_env
```

Reference in training config:
```toml
[[orchestrator.env]]
id = "my-env"
args = { difficulty = "hard" }
```

The `id` must match the package `name` in the environment's `pyproject.toml`.

## Endpoint Configuration

`configs/endpoints.py` defines model API endpoints for evaluation:

```python
ENDPOINTS = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "url": "http://0.0.0.0:8007/v1",
        "key": "local",
    },
}
```

## GPU Memory Guidelines

For a 7-8B model with LoRA on 80GB GPUs:

| Config | Peak Memory (approx) |
|--------|---------------------|
| No AC, seq_len=16384 | >80 GB (OOM) |
| AC freq=1, seq_len=16384 | ~65 GB |
| AC freq=1, seq_len=8192 | ~45 GB |

Options when OOMing on the trainer GPU:
1. Enable activation checkpointing (`[trainer.model.ac] freq = 1`)
2. Reduce `seq_len`
3. Use more trainer GPUs (split inference/trainer differently)
4. Enable FSDP CPU offloading (`fsdp_cpu_offload = true` under `[trainer.model]`)

## Troubleshooting

### All-zero training metrics (Loss: 0, Grad Norm: 0)

All rollouts are failing. Check `outputs/logs/orchestrator.stdout` for errors. Common causes:
- Duplicate sampling parameters (e.g., `top_p` in both config and `extra_body`)
- Environment `load_environment` errors
- Inference server not ready

### `batch_size must be divisible by rollouts_per_example`

Adjust one of the two values so `batch_size % rollouts_per_example == 0`.

### `ValidationError: ac` expects dict not string

Use nested TOML table syntax:
```toml
# Wrong
[trainer.model]
ac = "full"

# Right
[trainer.model.ac]
freq = 1
```

### Method signature mismatches in custom environments

Verifiers updates may change parent class signatures. Check that overridden methods (`is_completed`, `update_tool_args`, `env_response`) match the current parent signatures. Use `@vf.stop` for custom stop conditions instead of overriding `is_completed`.

### Entropy: NaN

No valid tokens in the training batch. Usually caused by all rollouts failing (see all-zero metrics above).


# References

- [Prime-RL Docs](https://github.com/PrimeIntellect-ai/prime-rl/tree/main/docs)
- [Prime-RL Entrypoints](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/entrypoints.md)

