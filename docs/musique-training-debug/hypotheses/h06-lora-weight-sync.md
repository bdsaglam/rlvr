# Hypothesis H06: LoRA Weight Synchronization Issue

## Priority: HIGH

## Description

The verifiers library may have issues properly synchronizing LoRA adapter weights between the policy model and reference model during GRPO training. This could lead to incorrect KL divergence calculations and ineffective gradient updates, causing training to stagnate despite resolving other issues.

## Root Cause

**Potential Issues**:
1. **Reference Model Not Updated**: LoRA adapters might not be properly copied to reference model
2. **Weight Sharing Conflicts**: Policy and reference models might inadvertently share weights
3. **Gradient Flow Issues**: LoRA parameters might not receive proper gradients during backprop
4. **Adapter Merging Problems**: Issues when computing log probabilities with adapters

## Why It Explains Our Behavior

1. **Rewards Not Improving**: Even after fixing KL beta=0, rewards still fluctuate without improvement
2. **Training Appears Stable But Ineffective**: Gradients are reasonable but model doesn't learn
3. **KL Divergence Was Extreme**: 1e5 KL values suggest reference model wasn't tracking policy properly
4. **Full Parameter Training Works**: Other users report success with full parameter training

## Supporting Evidence

### From Experimental Results (2025-09-11)
- Setting KL beta=0 fixed gradient explosion but not learning
- KL divergence reached 1e5 with beta=0.04 (should be 0-10 normally)
- This suggests reference model was completely disconnected from policy

### From Verifiers Library Architecture
- GRPO requires careful management of policy vs reference models
- LoRA adds complexity to weight management
- Library may not fully support PEFT adapters

## Validation Methods

### Method 1: Test Full Parameter Training
**Experiment**:
```bash
python scripts/train_musique.py train \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --no-peft \  # Disable LoRA
    --kl-beta 0.0 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --max-steps 100
```

**Success Criteria**:
- If full parameter training shows improvement but LoRA doesn't, confirms LoRA issue
- Should see steady reward increase over baseline

### Method 2: Inspect LoRA Weight Sync in Verifiers
**Code Investigation**:
```python
# Check verifiers/trainers/grpo_trainer.py for:
# 1. How reference model is initialized with LoRA
# 2. How log probabilities are computed for both models
# 3. Whether adapters are properly handled in forward passes
```

### Method 3: Add LoRA Diagnostic Logging
**Code Addition**:
```python
# In training loop, log adapter weight statistics
if hasattr(model, 'peft_modules'):
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            wandb.log({
                f"lora/{name}_mean": param.data.mean().item(),
                f"lora/{name}_std": param.data.std().item(),
                f"lora/{name}_grad_norm": param.grad.norm().item() if param.grad is not None else 0
            })
```

### Method 4: Manual Weight Sync Test
**Code Fix Attempt**:
```python
# Force manual sync of LoRA weights to reference model
if self.ref_model and hasattr(self.model, 'peft_modules'):
    # Copy LoRA adapter weights from policy to reference
    policy_state = self.model.state_dict()
    ref_state = self.ref_model.state_dict()
    for key in policy_state:
        if 'lora' in key.lower():
            ref_state[key] = policy_state[key].clone()
    self.ref_model.load_state_dict(ref_state, strict=False)
```

## Invalidation Criteria

This hypothesis is **invalidated** if:
1. Full parameter training has the same learning issues as LoRA
2. LoRA weights are properly synchronized in verifiers code review
3. Adding manual weight sync doesn't improve training
4. Other LoRA-based models train successfully with same setup

## Interaction with Other Hypotheses

- **H03 (KL Divergence)**: LoRA sync issues would explain extreme KL values
- **Partially explains H03 results**: Why fixing KL beta didn't fully resolve training
- **Independent of H01, H02**: This is a separate implementation issue

## Current Status: PENDING INVESTIGATION

## Experimental Priority

**Test Order**: 1 (HIGH - test immediately after documenting)
**Reason**:
- Explains remaining issues after H03 partial fix
- Easy to test with `--no-peft` flag
- Could be blocking all LoRA-based training

## Recommended Actions

1. **Immediate**: Test with `--no-peft` to confirm if LoRA is the issue
2. **If confirmed**: 
   - Review verifiers library LoRA implementation
   - Consider PR to fix weight synchronization
   - Document workaround for other users
3. **If not confirmed**: 
   - Move to H01 (Advantage Scaling) or H02 (Token Masking)
   - LoRA may be fine, issue lies elsewhere

## Notes

- This is a newly identified hypothesis based on experimental evidence
- May affect all LoRA-based training with verifiers library
- Could be a critical bug affecting many users
- Consider opening GitHub issue if confirmed

---

**Next Action**: Run comparison test between LoRA and full parameter training with all other settings identical.