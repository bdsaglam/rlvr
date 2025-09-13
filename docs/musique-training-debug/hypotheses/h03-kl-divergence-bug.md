# Hypothesis H03: KL Divergence Formula Bug

## Priority: MEDIUM

## Description

The KL divergence computation in the GRPO trainer may have the wrong direction. The current formula uses `exp(ref_per_token_logps - per_token_logps)`, but for KL(policy || reference), it should be `exp(per_token_logps - ref_per_token_logps)`. This backwards KL penalty could encourage divergence instead of constraining it.

## Root Cause

**Code Location**: `verifiers/trainers/grpo_trainer.py:1231-1235`

```python
per_token_kl = (
    torch.exp(ref_per_token_logps - per_token_logps)  # ⚠️ Potentially backwards
    - (ref_per_token_logps - per_token_logps)
    - 1
)
```

**Mathematical Issue**: 
- Standard KL(P || Q) = Σ P(x) log(P(x)/Q(x))
- For policy P and reference Q: KL penalty should discourage P from diverging from Q
- Current formula may penalize convergence and reward divergence

## Why It Explains Our Behavior

1. **Policy Instability**: Backwards KL penalty pushes model away from good reference behavior
2. **Oscillation**: Model gets conflicting signals from reward and KL terms
3. **Beta Sensitivity**: Higher KL beta values make problem worse (observed in experiments)

## Validation Methods

### Method 1: Test with KL Beta = 0
**Experiment**:
```bash
python scripts/train_musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train[:100]" \
    --kl-beta 0.0 \
    --batch-size 8 --num-generations 8 \
    --max-steps 100
```

**Success Criteria**:
- If training improves with beta=0 but fails with beta>0, confirms KL issue
- Should see more stable reward progression

### Method 2: Fix KL Formula and Test
**Code Fix**:
```python
# In verifiers/trainers/grpo_trainer.py, change to:
per_token_kl = (
    torch.exp(per_token_logps - ref_per_token_logps)  # Fixed direction
    - (per_token_logps - ref_per_token_logps)
    - 1
)
```

**Test**: Run with small positive beta (0.01) and compare to current implementation

### Method 3: Log KL Components
**Code Addition**:
```python
if self.beta != 0.0:
    # ... compute KL ...
    if wandb.run:
        wandb.log({
            "debug/kl_mean": mean_kl.item(),
            "debug/kl_max": per_token_kl.max().item(),
            "debug/kl_min": per_token_kl.min().item(),
            "debug/ref_logp_mean": ref_per_token_logps.mean().item(),
            "debug/policy_logp_mean": per_token_logps.mean().item(),
        })
```

**Success Criteria**: 
- KL should be positive and reasonable in magnitude
- KL should increase when policy diverges from reference

### Method 4: Compare Against Standard Implementation
**Reference Check**: Compare with TRL or other RL libraries to verify formula

## Invalidation Criteria

This hypothesis is **invalidated** if:
1. Training with `--kl-beta 0.0` has the same issues
2. KL values logged are reasonable and positive
3. Formula matches standard implementations
4. Fixing the formula doesn't improve training

## Interaction with Other Hypotheses

- **H01 (Advantage Scaling)**: KL bug could amplify advantage scaling problems
- **If H01 is fixed**: KL bug becomes more important to test
- **If H01 not fixed**: May be hard to isolate KL effects

## Current Status: PARTIALLY VALIDATED ✅

## Experimental Results (2025-09-11)

### Key Finding: KL Beta=0 Resolves Exploding Gradients

**Experiment**:
- Model: Qwen/Qwen2.5-7B-Instruct with LoRA (r=16, alpha=32)
- Loss type: dr_grpo
- Initial KL beta: 0.04 (default)

**Observations**:
1. **With KL beta=0.04**: 
   - `train/kl` reached **~1e5** (100,000x normal)
   - `train/grad_norm` reached **~1e5** (100,000x normal)
   - Gradients being clipped from 1e5 to 0.1 (1,000,000x reduction!)

2. **With KL beta=0.0**:
   - Gradient explosion resolved immediately
   - `train/grad_norm` returned to reasonable values
   - Training became stable

### Critical Insight: Dr. GRPO Should Use Beta=0

Research confirms that **Dr. GRPO explicitly excludes KL divergence (β=0)** when using rule-based verifiers because:
- Rule-based rewards (like MuSiQue's exact match) provide accurate signal regardless of distribution shift
- KL constraint is unnecessary and potentially harmful with deterministic verifiers
- The original Dr. GRPO paper recommends β=0 for this setup

### Remaining Issue: Rewards Still Not Improving

Despite fixing gradient explosion, rewards still fluctuate without clear improvement. This suggests:
1. The KL issue was masking another underlying problem
2. Possible LoRA weight synchronization issues in verifiers library (see new hypothesis H06)

## Experimental Priority

**Test Order**: COMPLETED (partially validated)
**Status**: 
- ✅ Confirmed that KL beta should be 0 for dr_grpo with rule-based verifiers
- ✅ Fixed gradient explosion issue
- ⚠️ Did not fully resolve training effectiveness problem

## Notes

- The verifiers library may have additional issues with LoRA training
- Need to investigate weight synchronization between policy and reference models
- Consider testing full parameter training to isolate LoRA-specific issues

---

**Next Action**: Test with `--kl-beta 0.0` after H01 validation. If H01 is confirmed, fix it first, then test KL independently.