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

## Current Status: UNTESTED

## Experimental Priority

**Test Order**: 3 (medium priority)
**Reason**:
- Mathematical bug would be systematic and severe
- Easy to test with beta=0 experiment
- But advantage scaling (H01) is more likely to be primary cause
- Should test after H01 results

## Notes

- User reported trying beta=0 in some experiments, but may have had other issues
- Need to test in isolation after addressing higher-priority hypotheses
- Formula should be double-checked against literature/standard implementations

---

**Next Action**: Test with `--kl-beta 0.0` after H01 validation. If H01 is confirmed, fix it first, then test KL independently.