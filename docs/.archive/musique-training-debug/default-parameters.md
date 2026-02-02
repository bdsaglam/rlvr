# Default Parameters Reference

This document tracks all default parameter values in `scripts/train_musique.py` for reference during hypothesis testing.

## Model Arguments
- `model`: "Qwen/Qwen2.5-7B-Instruct"

## Dataset Arguments  
- `datasets_str`: "bdsaglam/musique,answerable,train"
- `eval_datasets_str`: None
- `noise_rate`: 1.0

## Tool Parameters
- `retriever`: "hybrid"

## Generation Parameters
- `max_prompt_length`: 4096
- `max_new_tokens`: 1024  
- `temperature`: 0.5

## Training Arguments
- `num_epochs`: 1
- `save_steps`: 100
- `batch_size`: 8 ✅ (already optimal for testing)
- `num_generations`: 8 ✅ (already optimal for testing)
- `gradient_accumulation_steps`: 8
- `bf16`: False

## RL Training Parameters  
- `kl_beta`: 0.04
- `scale_rewards`: **False** ✅ (already disabled by default!)
- `loss_type`: "grpo"
- `num_iterations`: 1

## LoRA Arguments
- `peft`: True
- `lora_r`: 16 ✅ 
- `lora_alpha`: 32 ✅
- `lora_dropout`: 0.05

## Optimizer Arguments
- `learning_rate`: **1e-5** ✅ (already good default)
- `lr_scheduler_type`: "constant_with_warmup"
- `warmup_steps`: 10
- `max_grad_norm`: 0.1 ✅
- `gradient_checkpointing`: True

## Logging Arguments
- `logging_steps`: **1** ✅ (already frequent logging)
- `log_completions`: True ✅
- `log_on_each_node`: False

## Evaluation Arguments
- `per_device_eval_batch_size`: None

## Checkpointing Arguments  
- `save_only_model`: False

## Output Arguments
- `output_dir`: "./outputs"
- `run_name`: None (auto-generated)
- `push_to_hub`: False
- `hub_model_id`: None

## WandB Arguments
- `report_to`: "wandb"

## Resume Training
- `resume_from_checkpoint`: None

---

## Key Insights for H01 Testing

Looking at the defaults, I was **wrong** about several suggestions:

### ✅ Good Defaults (Don't Need to Change)
- `batch_size: 8` - already good for testing
- `num_generations: 8` - already good for testing  
- `learning_rate: 1e-5` - already reasonable
- `logging_steps: 1` - already frequent
- `max_grad_norm: 0.1` - already set
- `scale_rewards: False` - **ALREADY DISABLED BY DEFAULT!**

### 🚨 Critical Discovery: scale_rewards is False by Default!

This means **H01 might already be invalidated** - if the default is `scale_rewards=False` and you're still seeing the oscillation problem, then advantage scaling instability is NOT the root cause.

### 📝 Revised H01 Command

Since the defaults are already good for testing, your command only needs:

```bash
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train" \
    --model $MODEL \
    --bf16 \
    --num-epochs 3 \  # Add this for longer observation
    # Everything else uses good defaults!
    2>&1 | tee outputs/h01-baseline-$(date +%s).log
```

### 🔍 H01 Status Update

Since `scale_rewards=False` is the default, we need to **reframe H01**:
- **Test 1**: Confirm current behavior with defaults (should still show oscillation)
- **Test 2**: Try with `--scale-rewards` enabled to see if it makes things worse
- **If oscillation happens even with scaling disabled**: H01 is invalidated, move to H02

This is a good catch - it changes our testing priority!