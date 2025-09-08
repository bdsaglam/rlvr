# Hypothesis H02: Token Masking Errors

## Priority: HIGH

## Description

The model might be training on the wrong tokens due to incorrect attention masking or loss computation. In multi-turn tool-based conversations, if masking is incorrect, the model could be:
1. Training to predict tool responses (which it shouldn't)
2. Not training on its own assistant responses  
3. Computing gradients on environment/system tokens instead of model outputs

## Root Cause

**Potential Issues**:
1. **Environment Response Masking**: `mask_env_responses = True` in training args might not be implemented correctly
2. **Multi-turn Masking**: Complex conversation structure with tool calls may confuse token boundaries
3. **Completion vs Chat Format**: Mismatch between how completions are formatted and how loss is computed

## Why It Explains Our Behavior

1. **Gradients on Wrong Tokens**: Model receives gradient signal from predicting tool outputs, not reasoning
2. **No Learning Signal**: Real assistant responses get masked out, so no useful gradient signal
3. **Baseline Performance Maintained**: Model doesn't get worse because it's not really being updated on the right tokens

## Validation Methods

### Method 1: Inspect Token Boundaries
**Code Addition** to training script:
```python
def log_token_analysis(input_ids, attention_mask, labels, tokenizer):
    """Log what tokens are being trained on vs masked."""
    if wandb.run and random.random() < 0.01:  # Log 1% of batches
        for i in range(min(2, len(input_ids))):  # First 2 examples
            tokens = tokenizer.convert_ids_to_tokens(input_ids[i])
            mask = attention_mask[i] if attention_mask is not None else [1] * len(tokens)
            label_mask = (labels[i] != -100) if labels is not None else [1] * len(tokens)
            
            # Log which tokens are being trained on
            training_tokens = [
                f"{tok}({'✓' if label_mask[j] else '✗'})" 
                for j, tok in enumerate(tokens) if mask[j]
            ]
            
            wandb.log({
                f"debug/tokens_sample_{i}": " ".join(training_tokens[:50])  # First 50 tokens
            })
```

**Success Criteria**: Verify that:
- Only assistant response tokens are trained (not tool responses)
- Tool calls and environment responses are properly masked
- The model is actually training on reasoning content

### Method 2: Loss Distribution Analysis  
**Code Addition**:
```python
def analyze_loss_distribution(per_token_losses, completion_mask, tokenizer, input_ids):
    """Analyze where loss is coming from."""
    if wandb.run:
        # Compute loss statistics
        active_losses = per_token_losses[completion_mask.bool()]
        
        wandb.log({
            "debug/active_tokens_count": completion_mask.sum().item(),
            "debug/total_tokens_count": completion_mask.numel(),
            "debug/loss_mean": active_losses.mean().item(),
            "debug/loss_std": active_losses.std().item(),
            "debug/loss_max": active_losses.max().item(),
        })
```

### Method 3: Manual Completion Inspection
**Experiment**: 
```bash
# Add detailed logging of completions vs labels
python scripts/train_musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train[:20]" \
    --batch-size 2 --num-generations 4 \
    --max-steps 5 --logging-steps 1 \
    --log-completions true
```

**Manual Check**:
- Look at logged completions in WandB
- Verify format matches expected structure  
- Check if environment responses are appearing in completions
- Ensure assistant messages have proper boundaries

### Method 4: Compare Single-Turn vs Multi-Turn
**Experiment**: Create a simplified single-turn version of the task to isolate masking issues:
```python
# Test with completion-format instead of chat-format
# If single-turn works but multi-turn doesn't, confirms masking issue
```

## Invalidation Criteria

This hypothesis is **invalidated** if:
1. Token analysis shows correct masking (only assistant tokens trained)
2. Loss distribution is reasonable and focused on expected content
3. Manual inspection shows proper completion boundaries
4. Single-turn format has the same issues

## Diagnostic Questions

1. **Are we training on the right tokens?** 
   - Check what tokens have loss computed vs masked out
2. **Are tool responses properly excluded?**
   - Verify tool response tokens are masked with -100 labels  
3. **Is the conversation structure correct?**
   - Check message boundaries and role assignments
4. **Does completion format match expected structure?**
   - Verify <think>, <cite>, <answer> tags are in completions, not labels

## Current Status: UNTESTED

## Experimental Priority

**Test Order**: 2 (high priority)
**Reason**:
- Token masking bugs are subtle but devastating
- Would explain lack of learning despite apparently correct setup
- Multi-turn tool conversations are complex and error-prone
- Easy to add diagnostic logging

## Related Hypotheses

- H05: Format Parsing Failures (related to completion structure)
- H07: Multi-turn Conversation Issues (specific case of this)

---

**Next Action**: Add token boundary logging to training script and run diagnostic experiment.