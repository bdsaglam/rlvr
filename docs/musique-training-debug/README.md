# MuSiQue RL Training Debug Framework

A systematic approach to debugging why MuSiQue RL training shows no improvement despite decent baseline performance.

## 📋 Quick Status Overview

**Problem**: 7B models trained on MuSiQue with GRPO show no improvement over baseline (~60-70% reward) despite extensive hyperparameter tuning.

**Current Status**: Framework created, systematic testing in progress.

**Next Action**: [See Testing Roadmap](#testing-roadmap)

## 📁 Framework Structure

```
docs/musique-training-debug/
├── README.md                           # This file - overview and roadmap  
├── 00-problem-statement.md             # Complete problem context
├── 01-experiments-log.md               # All attempted experiments
├── hypotheses/                         # Individual hypothesis files
│   ├── h01-advantage-scaling-instability.md    # [HIGH] Numerical instability in advantages
│   ├── h02-token-masking-errors.md             # [HIGH] Training on wrong tokens
│   ├── h03-kl-divergence-bug.md                # [MED]  ✅ PARTIALLY VALIDATED - KL beta=0 fixes gradients
│   ├── h04-context-truncation.md               # [MED]  Prompt length limits
│   ├── h05-reward-signal-degradation.md        # [LOW]  Weak reward signal
│   ├── h06-lora-weight-sync.md                 # [HIGH] LoRA adapter synchronization issues
│   └── ...                                     # Additional hypotheses as needed
├── experimental-results/               # Test results and data
└── fixes/                             # Validated fixes and patches
```

## 🎯 Testing Roadmap

### Phase 1: High-Priority Hypotheses (Test These First)

#### H06: LoRA Weight Sync Issue 🔄 **[CRITICAL - TEST IMMEDIATELY]**
- **Issue**: LoRA adapters not properly synchronized between policy/reference models
- **Test**: Compare `--no-peft` (full param) vs LoRA training
- **Time**: 1-2 hours for comparison
- **Impact**: Could explain why rewards don't improve despite fixing gradients
- **Status**: 🆕 Newly identified from H03 results

#### H01: Advantage Scaling Instability ⚡ **[HIGH - TEST AFTER H06]**
- **Issue**: Division by small std values creates massive advantages  
- **Test**: `--scale-rewards false` with diagnostic logging
- **Time**: 1-2 hours for quick validation
- **Impact**: May be secondary to LoRA issues

#### H02: Token Masking Errors 🎭 **[HIGH - TEST AFTER H01]**  
- **Issue**: Training on environment tokens instead of assistant responses
- **Test**: Add token boundary logging, inspect what's being trained
- **Time**: 2-3 hours for diagnostics  
- **Impact**: Could explain lack of learning signal

### Phase 2: Medium-Priority Hypotheses (Test If Phase 1 Doesn't Resolve)

#### H03: KL Divergence Bug 🧮
- **Issue**: Backwards KL formula may encourage divergence
- **Test**: Compare `--kl-beta 0.0` vs fixed formula
- **Prerequisites**: Complete H01 first (interact with advantage scaling)

#### H04: Context Truncation ✂️  
- **Issue**: Important documents truncated due to length limits
- **Test**: Increase `--max-prompt-length` and log context stats
- **Resource Cost**: Higher memory usage

### Phase 3: Lower-Priority Hypotheses (Last Resort)

#### H05: Reward Signal Issues 📊
- **Issue**: Noisy or conflicting reward components
- **Test**: Single-component rewards, distribution analysis  
- **Note**: Less likely given baseline performance

## 🧪 Systematic Testing Protocol

### Before Each Experiment
1. **Document Hypothesis**: Which one you're testing and why
2. **Set Success Criteria**: What result would confirm/refute the hypothesis
3. **Use Minimal Setup**: Small dataset, short training for faster feedback
4. **Enable Diagnostics**: Add relevant logging for the hypothesis

### Standard Test Command Template
```bash
# Quick validation experiment (use this format)
python scripts/train_musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train[:100]" \
    --batch-size 8 --num-generations 8 \
    --max-steps 50 --logging-steps 1 \
    --output-dir ./outputs/debug-h0X-test \
    [HYPOTHESIS-SPECIFIC-FLAGS]
```

### After Each Experiment
1. **Document Results**: Update hypothesis file with findings
2. **Update Status**: Mark as Validated/Invalidated/Inconclusive  
3. **Next Steps**: Based on results, which hypothesis to test next
4. **Evidence Quality**: Rate confidence in results (High/Medium/Low)

## 📊 Result Tracking

### Hypothesis Status Tracking
- ✅ **Validated**: Confirmed as root cause, fix implemented
- ❌ **Invalidated**: Ruled out by experimental evidence
- 🧪 **Testing**: Currently being investigated  
- ⏳ **Pending**: Not yet tested
- 🤔 **Inconclusive**: Results unclear, needs more testing

### Current Status Summary
| Hypothesis | Priority | Status | Confidence | Notes |
|------------|----------|--------|------------|-------|
| H01: Advantage Scaling | HIGH | ⏳ Pending | - | Test after H06 |
| H02: Token Masking | HIGH | ⏳ Pending | - | Test after H01 |
| H03: KL Divergence | MED | ✅ Partial | HIGH | KL beta=0 fixed gradients, not rewards |
| H04: Context Truncation | MED | ⏳ Pending | - | Memory intensive |
| H05: Reward Signal | LOW | ⏳ Pending | - | Last resort |
| H06: LoRA Weight Sync | HIGH | 🧪 Testing | - | **NEW - Test immediately with --no-peft** |

## 🔧 Diagnostic Tools

### Essential Logging Additions
Add these to your training script for better debugging:

```python
# Advantage scaling diagnostics
def log_advantage_stats(advantages, rewards):
    if wandb.run:
        wandb.log({
            "debug/advantage_max": advantages.abs().max().item(),
            "debug/advantage_mean": advantages.abs().mean().item(),
            "debug/reward_std": rewards.std().item(),
        })

# Token masking diagnostics  
def log_token_boundaries(input_ids, labels, tokenizer, sample_rate=0.01):
    if wandb.run and random.random() < sample_rate:
        # Log which tokens are being trained vs masked
        # ... (see H02 for full implementation)

# Context length diagnostics
def log_context_stats(input_ids, max_length):
    if wandb.run:
        lengths = [len(ids) for ids in input_ids]
        truncated = sum(1 for l in lengths if l >= max_length - 10)
        wandb.log({
            "debug/context_length_mean": np.mean(lengths),
            "debug/context_truncation_rate": truncated / len(lengths),
        })
```

## 🚀 Getting Started

**Immediate next steps:**

1. **Read H01 (Advantage Scaling)** - most likely culprit
2. **Run the H01 validation experiment** - should take 1-2 hours
3. **Based on H01 results**, either:
   - If confirmed: Implement fix and retest  
   - If not: Move to H02 (Token Masking)
4. **Update this README** with your findings

## 📝 Contributing to This Framework

When adding new hypotheses:
1. Use the template from existing hypothesis files
2. Include clear validation/invalidation criteria
3. Estimate resource requirements and testing time
4. Update the status table in this README
5. Consider interactions with existing hypotheses

## 🏆 Success Criteria

This debugging effort succeeds when:
- ✅ Training shows consistent reward improvement over baseline
- ✅ Improvement is reproducible across multiple runs  
- ✅ We can explain why the fix works
- ✅ Fix is validated on full-scale training (not just debug runs)

---

**Remember**: Test systematically, document everything, and don't skip ahead. The methodical approach will save time in the long run.