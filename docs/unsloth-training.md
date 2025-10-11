# Unsloth Training

This guide explains how to use the Unsloth-optimized training script for faster and more memory-efficient training.

## Overview

The `train_musique_unsloth.py` script uses [Unsloth](https://github.com/unslothai/unsloth) to achieve:
- 2x faster training speeds
- Reduced memory usage
- Optimized gradient checkpointing
- Better LoRA implementation

## Installation

Install Unsloth:

```bash
pip install unsloth
```

Or for specific CUDA versions, see [Unsloth installation guide](https://github.com/unslothai/unsloth#installation).

## Usage

The Unsloth training script works identically to the standard training script:

```bash
python scripts/train_musique_unsloth.py train \
    --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 8 \
    --num-generations 8 \
    --learning-rate 1e-5 \
    --max-steps 100
```

### Key Differences from Standard Training

1. **Model Loading**: Uses `FastLanguageModel.from_pretrained()` instead of standard HF loaders
2. **LoRA Application**: PEFT is applied via `FastLanguageModel.get_peft_model()` with Unsloth optimizations
3. **Gradient Checkpointing**: Automatically uses Unsloth's optimized implementation (`use_gradient_checkpointing="unsloth"`)
4. **LoRA Alpha**: Automatically doubled (2x) for faster training as recommended by Unsloth

## Architecture

### How It Works

The training script integrates Unsloth with the standard `EnhancedGRPOTrainer`:

```python
from unsloth import FastLanguageModel
from rlvr.trainers import EnhancedGRPOTrainer

# Load and configure with Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    max_seq_length=8192,
    load_in_4bit=False,
    fast_inference=True,
)

# Apply PEFT with Unsloth optimizations
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    use_gradient_checkpointing="unsloth",
)

# Use standard trainer with pre-configured model
trainer = EnhancedGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=env,
    args=training_args,
    peft_config=None,  # Already applied by Unsloth
)
```

The integration works by:
1. Pre-applying PEFT via Unsloth's `FastLanguageModel.get_peft_model()`
2. Passing `peft_config=None` to skip duplicate PEFT application
3. GRPOTrainer automatically detects the model already has PEFT and uses it correctly
4. All GRPO training logic and W&B logging work unchanged

## Configuration Options

All standard training options are supported. Notable defaults:

- `--gradient-checkpointing`: Automatically uses Unsloth's implementation
- `--lora-alpha`: Doubled internally for faster training
- `--bf16`: Recommended for best performance

## Performance Tips

1. **Memory Usage**: Adjust `gpu_memory_utilization` if you encounter OOM errors
2. **LoRA Rank**: Higher ranks (32, 64) work well with Unsloth's optimizations
3. **Batch Size**: You may be able to use larger batch sizes with Unsloth
4. **Mixed Precision**: Use `--bf16` for A100/H100 GPUs

## Troubleshooting

### ImportError: unsloth

Install Unsloth:
```bash
pip install unsloth
```

### Flash Attention Version Mismatch

Unsloth handles this automatically and will patch the system to work with your installed version.

### OOM Errors

Try reducing:
- `--batch-size`
- `--num-generations`
- `--max-prompt-length`
- Or adjust `gpu_memory_utilization` in the code (currently 0.9)

## References

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [GRPO Training Guide](./grpo-training.md)
