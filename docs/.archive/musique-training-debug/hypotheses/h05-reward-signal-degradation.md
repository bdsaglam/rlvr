# Hypothesis H05: Reward Signal Degradation

## Priority: MEDIUM-LOW

## Description

Despite baseline success, the reward signal may have subtle quality issues that prevent effective learning. The combined reward function has multiple components with different scales and noise levels, potentially creating a weak or misleading gradient signal.

## Root Cause

**Potential Issues**:
1. **Reward Noise**: Individual reward components may be noisy, overwhelming the true signal
2. **Component Conflicts**: Different reward components (EM, F1, retrieval, citation) may give conflicting signals
3. **Sparse Differentiation**: Most completions get similar rewards (0.6-0.7), providing little contrast for learning
4. **Format Bias**: Format reward may dominate other signals, causing model to optimize for format over content quality

## Why It Explains Our Behavior

1. **Weak Gradient Signal**: Small reward differences don't provide strong learning signal
2. **Local Optima**: Model stays in "good enough" region without strong incentive to improve
3. **Component Interference**: Conflicting reward components average out to no clear direction

## Validation Methods

### Method 1: Single Reward Component Testing
**Experiment**: Test with only one reward component at a time:
```bash
# Test with only exact match reward
python scripts/modify_reward.py --only-exact-match
python scripts/train_musique.py train --datasets "bdsaglam/musique-mini,answerable,train[:100]"

# Test with only retrieval recall  
python scripts/modify_reward.py --only-retrieval-recall
python scripts/train_musique.py train --datasets "bdsaglam/musique-mini,answerable,train[:100]"
```

**Success Criteria**: If training improves with single components, confirms component interference

### Method 2: Reward Distribution Analysis
**Code Addition**:
```python
def analyze_reward_distribution(rewards, reward_components):
    """Analyze reward signal quality."""
    if wandb.run:
        rewards_tensor = torch.tensor(rewards)
        
        wandb.log({
            "debug/reward_mean": rewards_tensor.mean().item(),
            "debug/reward_std": rewards_tensor.std().item(),
            "debug/reward_range": (rewards_tensor.max() - rewards_tensor.min()).item(),
            "debug/reward_entropy": -torch.sum(F.softmax(rewards_tensor, dim=0) * F.log_softmax(rewards_tensor, dim=0)).item(),
        })
        
        # Log component correlations
        for comp_name, comp_values in reward_components.items():
            comp_tensor = torch.tensor(comp_values)
            correlation = torch.corrcoef(torch.stack([rewards_tensor, comp_tensor]))[0,1]
            wandb.log({f"debug/reward_correlation_{comp_name}": correlation.item()})
```

### Method 3: Reward Scaling Experiments
**Experiment**: Test different reward component weights:
```python
# Emphasize exact match more heavily
pairs = [
    (em_reward, 10.0),      # Increased weight
    (_f1_reward, 0.9),
    (retrieval_recall, 1.0),
    (retrieval_precision, 0.4),
    (citation_score, 0.6),
    (format_score, 0.1),
]
```

### Method 4: Reward Difference Analysis  
**Code Addition**:
```python
def analyze_reward_differences(batch_rewards, num_generations):
    """Analyze how much rewards differ within each prompt group."""
    rewards_grouped = batch_rewards.view(-1, num_generations)
    within_group_std = rewards_grouped.std(dim=1).mean()
    between_group_std = rewards_grouped.mean(dim=1).std()
    
    if wandb.run:
        wandb.log({
            "debug/within_group_reward_std": within_group_std.item(),
            "debug/between_group_reward_std": between_group_std.item(),
            "debug/reward_contrast_ratio": (within_group_std / (between_group_std + 1e-8)).item(),
        })
```

**Success Criteria**: 
- `within_group_std` should be sufficient for learning (>0.1)
- If contrast ratio is very low, confirms weak signal issue

## Invalidation Criteria

This hypothesis is **invalidated** if:
1. Reward distribution shows good variance and contrast
2. Single-component rewards don't improve training
3. Reward correlations are reasonable (components not conflicting)
4. Higher-priority hypotheses (H01-H02) explain the behavior better

## Alternative Reward Formulations

If validated, consider:
1. **Binary Rewards**: Use only exact match (0 or 1)
2. **Staged Rewards**: Weight format heavily early, then shift to content
3. **Sparse Rewards**: Only give rewards for significant improvements
4. **Curriculum Learning**: Start with easier examples, gradually increase difficulty

## Current Status: UNTESTED

## Experimental Priority

**Test Order**: 5 (lower priority)
**Reason**:
- Reward components seem to work (baseline performance indicates signal quality)
- More complex to test and debug than higher-priority issues
- Should test after ruling out advantage scaling and masking issues
- May be secondary effect rather than root cause

## Notes

- This is more of a "last resort" hypothesis if others don't pan out
- The fact that baseline model gets reasonable performance suggests reward signal isn't fundamentally broken
- But might explain plateau behavior if signal is too noisy/weak for further improvement

---

**Next Action**: Test only after H01-H04 are explored. Focus on reward distribution analysis first.