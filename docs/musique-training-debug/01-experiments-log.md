# Experiment Log: MuSiQue RL Training Attempts

This document tracks all training experiments attempted, organized chronologically with key parameters and outcomes.

## Experiment Categories

### A. Qwen2.5-7B with LoRA
**Base Command Pattern**:
```bash
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --model Qwen/Qwen2.5-7B-Instruct
```

#### A1. High-Rank LoRA Experiments
- **LoRA Config**: r=64, alpha=128
- **Training**: batch_size=16, num_generations=8, grad_accum=8
- **Dataset**: `bdsaglam/musique,answerable,train`
- **Outcome**: No improvement observed

#### A2. Different LoRA Ranks
- **Configs Tested**: 
  - r=64, alpha=64
  - r=16, alpha=32  
  - r=8, alpha=16
- **Common Pattern**: Rewards fluctuate around baseline, no upward trend

#### A3. Reward Scaling + Loss Type Variations
- **Command**:
```bash
--scale-rewards \
--loss-type dr_grpo \
--lora-r 16 --lora-alpha 32 \
--batch-size 16 --num-generations 8 --gradient-accumulation-steps 2
```
- **Outcome**: Some experiments showed "reward stalls"

### B. No PEFT (Full Fine-tuning)
```bash
--no-peft \
--batch-size 8 \
--max-completion-length 1024
```
- **Issue**: OOM errors on 7B model
- **Models That Worked**: Smaller models only

### C. Different Models

#### C1. Llama3.1-8B-Instruct  
- **Special Requirement**: Custom chat template needed
```bash
--model meta-llama/Llama-3.1-8B-Instruct \
--no-peft \
--batch-size 8
```
- **Issue**: "ValueError: This model only supports single tool-calls at once!"

#### C2. Qwen3-8B
```bash
--model willcb/Qwen3-8B \
--no-peft \
--batch-size 16
```

#### C3. Qwen3-4B  
```bash
--model willcb/Qwen3-4B \
--data-parallel-size 2 \
--loss-type dr_grpo \
--lora-r 8 --lora-alpha 16
```

### D. Hyperparameter Sweeps

#### D1. Learning Rate Variations
- **Tested**: 1e-6, 1e-5, 5e-6
- **Pattern**: Very small rates (1e-6) may prevent meaningful updates

#### D2. KL Beta Variations
- **Tested**: 0.0, 0.01, 0.04
- **Including**: `--kl-beta 0` (no KL penalty)

#### D3. Generation Count
- **Standard**: 8 generations per prompt  
- **Tried**: 16 generations for more diverse sampling

#### D4. Gradient Clipping
- **Values**: 0.01, 0.1
- **With**: Different max_grad_norm settings

### E. Dataset Variations

#### E1. Mini Dataset
```bash
--datasets "bdsaglam/musique-mini,answerable,train"
```
- **Purpose**: Faster iteration, same behavior

#### E2. Sliced Dataset
```bash
--datasets "bdsaglam/musique,answerable,train[:256]"
```
- **Purpose**: Small scale debugging

#### E3. Different Splits
- **Train**: Various train splits tested
- **Eval**: Some experiments included evaluation datasets

### F. Infrastructure Variations

#### F1. DeepSpeed vs Model Replication
```bash
# With DeepSpeed ZeRO
--config-file configs/zero3.yaml

# Without DeepSpeed (model replication)
# (no config file)
```

#### F2. Different GPU Configurations
- **2 GPUs**: `--num-processes 2` 
- **3 GPUs**: `--num-processes 3`
- **Memory**: Varied gpu-memory-utilization (0.6-0.7)

### G. Stable Training Attempts (Recent)

#### G1. Conservative Settings
```bash
--bf16 \
--loss-type "grpo" \
--lora-r 16 --lora-alpha 32 \
--batch-size 16 --num-generations 8 --gradient-accumulation-steps 4 \
--scale-rewards \
--max-grad-norm 0.01 \
--learning-rate 1e-5
```

#### G2. Ultra-Conservative  
```bash
--loss-type "grpo" \
--scale-rewards \
--lora-r 8 --lora-alpha 16 \
--batch-size 8 --num-generations 16 \
--max-grad-norm 0.1 \
--learning-rate 1e-6
```

## Patterns Observed

### Consistent Across Experiments
1. **Baseline Performance**: ~60-70% combined reward from start
2. **No Improvement Trend**: Rewards stay flat throughout training
3. **Fluctuation**: Rewards vary but don't trend upward
4. **Successful Examples**: Individual high-quality completions exist

### Model-Specific Issues
- **Llama3.1**: Tool calling compatibility issues
- **Larger Models**: OOM with full fine-tuning
- **All Models**: Same core issue (no learning)

### Infrastructure Robustness
- **No Crashes**: Training runs complete successfully  
- **No OOM**: With appropriate batch sizes
- **Gradient Flow**: No NaN or explosion detected

## Key Insights from Experiments

1. **Problem is NOT**:
   - Infrastructure setup (multiple configs work)
   - Model choice (happens across models)  
   - Basic hyperparameters (wide range tested)
   - Dataset size (happens on mini and full)

2. **Problem MIGHT BE**:
   - Advantage computation (numerical issues)
   - Reward signal quality (despite baseline success)
   - Token masking (training on wrong tokens)
   - Subtle optimizer issues (gradients not updating correctly)

## Current Status

- **Total Experiments**: 20+ different configurations
- **Time Invested**: Multiple weeks of training cycles
- **Next Direction**: Systematic hypothesis testing (this framework)
- **Parallel Test**: GSM8K dataset to isolate MuSiQue-specific issues

---

*Updated as new experiments are conducted. Each major experiment should be documented here with key parameters and outcomes.*