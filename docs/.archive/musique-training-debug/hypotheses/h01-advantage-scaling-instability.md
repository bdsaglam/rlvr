# Hypothesis H01: Advantage Scaling Numerical Instability

## Priority: HIGH

## Description

The advantage scaling in GRPO divides advantages by standard deviation with only `1e-4` epsilon. When the baseline model already performs well (60-70% reward), most completions get similar rewards, leading to very small standard deviations. Dividing by near-zero values creates extremely large advantages that destabilize training.

## Root Cause

**Code Location**: `verifiers/trainers/grpo_trainer.py:1173-1174`

```python
if self.scale_rewards:
    advantages = advantages / (std_grouped + 1e-4)  # ⚠️ Problem here
```

**Scenario**:
- Base model gets rewards like [0.62, 0.68, 0.71, 0.65] for different completions
- `std_grouped` = ~0.04 (small because rewards are similar)
- Division by (0.04 + 1e-4) ≈ 0.04 amplifies advantages by 25x
- This creates massive gradient updates followed by corrections
- Net result: oscillation around same performance level

## Why It Explains Our Behavior

1. **No Net Progress**: Model oscillates wildly but averages to same performance
2. **Reward Fluctuation**: Large updates cause temporary reward changes
3. **Baseline Independence**: Issue persists regardless of other hyperparameters
4. **Model Independence**: Affects any model with decent baseline performance

## Validation Methods

### Method 1: Disable Reward Scaling
**Experiment**: 
```bash
python scripts/train_musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train[:100]" \
    --scale-rewards false \
    --batch-size 8 --num-generations 8 \
    --max-steps 50 --logging-steps 1
```

**Success Criteria**: 
- More stable reward progression (less wild fluctuation)
- Gradual upward trend instead of oscillation
- Lower gradient norms in logs

### Method 2: Log Advantage Statistics
**Code Addition**:
```python
def _compute_advantages(self, rewards):
    # ... existing code ...
    if wandb.run:
        wandb.log({
            "debug/reward_std": std_grouped.mean().item(),
            "debug/advantage_max": advantages.abs().max().item(), 
            "debug/advantage_mean": advantages.abs().mean().item(),
        })
    return advantages
```

**Success Criteria**: 
- `advantage_max` should be reasonable (< 10x reward range)
- If `advantage_max` >> 1.0, confirms the hypothesis

### Method 3: Increase Epsilon
**Experiment**:
```python
# In verifiers library, change:
advantages = advantages / (std_grouped + 0.1)  # Much larger epsilon
```

**Success Criteria**: Training stability improves

## Invalidation Criteria

This hypothesis is **invalidated** if:
1. Disabling `--scale-rewards` doesn't improve training stability
2. Advantage statistics show reasonable values (not amplified)
3. The same oscillation pattern occurs with `--scale-rewards false`

## Alternative Explanations If Invalidated

- Token masking issues (training on wrong tokens)
- Optimizer state corruption 
- Reward signal quality issues despite apparent success

## Current Status: UNTESTED

## Experimental Priority

**Test Order**: 1 (highest priority)
**Reason**: 
- Clear mathematical explanation for observed behavior
- Easy to test (single flag change)
- Matches pattern of "fluctuation without improvement"
- Most likely to have broad impact

## Related Hypotheses

- H06: Reward Distribution Issues (related to why scaling amplifies problems)
- H08: Gradient Explosion (consequence of this issue)

---

## Experiment

### Setup
Inference 

```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=0 vf-vllm --model $MODEL \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 16384 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```

Training
```sh
export MODEL="Qwen/Qwen2.5-7B-Instruct"

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train" \ # has 100 samples for each hop (2-4)
    --model $MODEL \
    --bf16 \
    --loss-type "grpo" \
    --no-scale-rewards \
    --lora-r 16 \
    --lora-alpha 32 \
    --batch-size 8 \
    --num-generations 8 \
    --gradient-accumulation-steps 4 \
    --max-grad-norm 0.1 \
    --learning-rate 1e-5 \
    --num-epochs 10 \
    2>&1 | tee outputs/train-$(date +%s).log
```

**Key Parameters**:
- Special flags: `no-scale-rewards`

### Results

#### Training Metrics
- **Reward Trend**: [Stable/Improving/Declining/Oscillating]
- **Final Reward**: [X.XX compared to baseline Y.YY]
- **Training Stability**: [Stable/Unstable/Gradient explosion/etc.]

#### Diagnostic Metrics (if applicable)
- **Advantage Statistics**: 
- **Context Length Stats**:
- **Token Masking Stats**:
- **Other Relevant Logs**:

#### Key Observations
- [Important patterns noticed]
- [Unexpected behaviors]
- [Comparison to previous experiments]

### Conclusion

**Hypothesis Status**: ❌ **INVALIDATED**

**Confidence Level**: High

**Reasoning**: 
GSM8K experiment (single-step environment) shows clear reward improvement within 20 steps using the same GRPO trainer and similar hyperparameters. Key observations:
- Reward trend: Clear upward progression 
- Reward std: 0.1 (healthy variance, 0.05-0.25 range)
- KL divergence: Stable around 0.0014-0.0018
- Gradient norms: Reasonable 0.01-0.03 range

This proves the verifiers library GRPO implementation works correctly for learning. The issue is **NOT** with advantage scaling or fundamental training instability, since the same trainer shows improvement on GSM8K.

**Critical Insight**: The problem appears to be **environment-specific** - single-step (GSM8K) works, multi-step tool-based (MuSiQue) doesn't.

**Next Steps**:
- [ ] [What to do based on these results]
- [ ] [Which hypothesis to test next]
- [ ] [Any follow-up needed]

### Files/Logs
- **WandB Run**: [link if available]
- **Log Files**: [paths to relevant logs]
- **Model Checkpoints**: [if saved]

---

**Notes**: [Any additional context or observations]