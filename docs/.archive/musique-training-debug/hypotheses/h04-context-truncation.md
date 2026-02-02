# Hypothesis H04: Context Length Truncation

## Priority: MEDIUM

## Description

MuSiQue prompts with retrieved documents may frequently exceed the maximum context length, causing critical information to be truncated. If the model never sees the relevant supporting documents due to truncation, it cannot learn to improve its reasoning, leading to performance plateau.

## Root Cause

**Context Length Constraints**:
- `max_prompt_length = 4096` (from training script)
- MuSiQue documents can be long (title + full paragraph text)
- Multiple retrieved documents per question (2+ documents typical)
- System prompt + question + tool responses + documents can exceed 4K tokens

**Truncation Behavior**: 
- When context exceeds limit, important supporting documents may be cut off
- Model gets inconsistent training signal (sometimes sees answer, sometimes doesn't)
- Performance plateaus because model can't reliably access information needed to improve

## Why It Explains Our Behavior

1. **Inconsistent Learning Signal**: Model sometimes has access to answer, sometimes doesn't
2. **Performance Ceiling**: Can't improve beyond what's achievable with truncated context
3. **Baseline Maintenance**: Short questions still work, so performance doesn't degrade
4. **Tool Call Confusion**: Model may learn tools don't provide useful information (due to truncation)

## Validation Methods

### Method 1: Log Context Length Statistics
**Code Addition**:
```python
def log_context_stats(input_ids, tokenizer, max_length):
    """Log context length statistics to identify truncation."""
    if wandb.run:
        lengths = [len(ids) for ids in input_ids]
        truncated_count = sum(1 for l in lengths if l >= max_length - 10)  # Near limit
        
        wandb.log({
            "debug/context_length_mean": np.mean(lengths),
            "debug/context_length_max": max(lengths),
            "debug/context_length_truncated": truncated_count,
            "debug/context_length_truncation_rate": truncated_count / len(lengths),
        })
```

**Success Criteria**: 
- If >20% of examples are near max_length, confirms truncation issue
- Should see correlation between truncation rate and reward variance

### Method 2: Increase Context Length
**Experiment**:
```bash
python scripts/train_musique.py train \
    --datasets "bdsaglam/musique-mini,answerable,train[:100]" \
    --max-prompt-length 8192 \  # Double the context length
    --max-new-tokens 1024 \
    --batch-size 4 \  # Reduce batch size due to memory
    --num-generations 4
```

**Success Criteria**: 
- If training improves with longer context, confirms truncation was the issue
- Should see better retrieval recall metrics

### Method 3: Analyze Document Retrieval Success
**Code Addition**:
```python
def analyze_retrieval_success(completion, info):
    """Check if retrieved documents contain supporting information."""
    supporting_docs = [doc for doc in info["docs"] if doc["is_supporting"]]
    retrieved_ids = extract_all_retrieved_doc_ids(completion)
    
    # Check how often supporting docs are successfully retrieved
    retrieval_success = len(set(retrieved_ids) & set(d["id"] for d in supporting_docs))
    
    return {
        "supporting_docs_available": len(supporting_docs),
        "supporting_docs_retrieved": retrieval_success,
        "retrieval_success_rate": retrieval_success / len(supporting_docs) if supporting_docs else 1.0
    }
```

### Method 4: Prompt Length Pre-filtering
**Experiment**: Filter dataset to only include examples that fit comfortably in context:
```python
def filter_by_prompt_length(example, max_length=3000):
    """Filter examples by estimated prompt length."""
    # Rough estimate: question + system prompt + tool setup
    estimated_length = len(example["question"]) * 1.3 + 500  # rough token estimate
    return estimated_length < max_length
```

**Success Criteria**: If filtered dataset shows improvement, confirms context issue

## Invalidation Criteria

This hypothesis is **invalidated** if:
1. Context length analysis shows most examples well under limit
2. Increasing context length doesn't improve training
3. Document retrieval success rate is high even with current limits
4. Pre-filtered short examples have same training issues

## Diagnostic Questions

1. **What's the typical prompt length after tool responses?**
2. **How often do supporting documents get truncated out?**  
3. **Is there correlation between prompt length and reward?**
4. **Do longer context limits improve retrieval metrics?**

## Current Status: UNTESTED

## Experimental Priority

**Test Order**: 4 (medium priority after H01-H02)
**Reason**:
- Context truncation is a common cause of RL plateau in long-context tasks
- Relatively easy to test by increasing limits
- But requires more GPU memory, so test after cheaper hypotheses
- Could interact with other issues (mask higher priorities first)

## Memory Requirements

- Doubling context length roughly doubles memory usage
- May need to reduce batch size or use gradient checkpointing  
- Consider using model sharding if needed

---

**Next Action**: Add context length logging to training script. If truncation rate >15%, prioritize testing increased context length.