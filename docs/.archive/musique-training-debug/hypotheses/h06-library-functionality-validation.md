# Hypothesis H06: Library Functionality Validation

## Priority: HIGH

## Description

Before debugging MuSiQue-specific issues, we need to validate that the verifiers GRPO library actually works for RL training on any environment. If the library itself is broken, no amount of MuSiQue debugging will help.

## Root Cause

**Potential Issues**:
- GRPO trainer implementation bugs in verifiers library
- Fundamental issues with advantage computation, KL divergence, or loss calculation
- Environment-trainer interface problems
- Gradient flow issues in the training loop

## Why It Explains Our Behavior

If the library is fundamentally broken:
1. **No Learning Signal**: Training would fail across all environments
2. **Universal Problem**: Issue wouldn't be specific to MuSiQue
3. **Gradient Issues**: Mathematical bugs would prevent any improvement

## Validation Method

**Experiment**: Test GRPO training on GSM8K (single-step environment) with same trainer

Install environment

```sh
vf-install gsm8k --from-repo
```


Inference 

```sh
export MODEL="Qwen/Qwen2.5-3B-Instruct"

CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --enforce-eager
```

Training
```sh
export MODEL="Qwen/Qwen2.5-3B-Instruct"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_gsm8k.py train \
    --model $MODEL \
    --loss-type "grpo" \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 16 \
    --num-generations 8 \
    --gradient-accumulation-steps 8 \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```

**Success Criteria**: 
- Clear reward improvement within 20-50 steps
- Stable gradient norms and KL values  
- No oscillation or plateau behavior

## Results

### Experiment: GSM8K Training
**Date**: Current  
**WandB**: https://wandb.ai/bdsaglam/rlvr-debug/runs/7wp8qn2s

#### Training Metrics
- **Reward Trend**: ✅ **Clear upward progression** within 20 steps
- **Reward Std**: 0.1 (healthy variance, range 0.05-0.25)
- **KL Divergence**: Stable 0.0014-0.0018 range
- **Gradient Norms**: Reasonable 0.01-0.03 range
- **Training Stability**: ✅ **Stable improvement, no wild oscillation**

#### Key Observations
- GRPO trainer works correctly for single-step environments
- Same hyperparameters that fail on MuSiQue succeed on GSM8K
- Library implementation is fundamentally sound
- Issue is **environment-specific**, not library-wide

## Conclusion

**Hypothesis Status**: ❌ **INVALIDATED** 

**Confidence Level**: High

**Reasoning**: 
GSM8K experiment proves the verifiers GRPO library works correctly. The trainer can learn effectively with proper reward progression, stable metrics, and reasonable gradient behavior. This rules out fundamental library bugs.

**Critical Discovery**: The issue is **MuSiQue environment-specific** or **multi-step tool interaction-specific**, not a general GRPO trainer problem.

## Next Steps Based on Results

1. ✅ **Library works** - focus on MuSiQue-specific issues
2. 🎯 **Test multi-step tool environment** - find simpler multi-step env to isolate tool interaction vs MuSiQue-specific issues
3. 🔍 **Prioritize H02 (Token Masking)** and **H07 (Multi-step Environment Issues)**

## Related Hypotheses

- **H07: Multi-step vs Single-step Environment Issues** (NEW - to be created)
- **H02: Token Masking Errors** (now higher priority)
- **H04: Context Truncation** (MuSiQue-specific issue)

---

**Impact**: This completely reframes the debugging approach - we now know the trainer works, so focus on environment-specific issues.


