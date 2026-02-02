# Hypothesis H07: Multi-step vs Single-step Environment Issues

## Priority: HIGH

## Description

The verifiers GRPO trainer works on single-step environments (GSM8K) but fails on multi-step tool-based environments (MuSiQue). This suggests the issue is related to multi-step interaction patterns, tool usage, or conversation state management rather than fundamental training problems.

## Root Cause

**Potential Issues**:
1. **Tool Interaction Complexity**: Multi-step environments require tool calls and response handling
2. **Conversation State**: Multi-turn conversations may confuse token boundaries
3. **Action-Observation Loops**: Complex state transitions between tool usage and reasoning
4. **Reward Attribution**: Rewards may not properly attribute to the right actions in multi-step sequences

## Why It Explains Our Behavior

**Single-step (GSM8K)**:
- Direct question → reasoning → answer format
- Clear token boundaries for training
- Simple reward attribution
- ✅ **Works fine** (confirmed via experiment)

**Multi-step (MuSiQue)**:
- Question → tool call → tool response → tool call → reasoning → answer
- Complex conversation structure with multiple turns
- Reward depends on entire multi-step sequence
- ❌ **No improvement** (observed pattern)

## Validation Methods

### Method 1: Test Another Multi-step Environment
**Experiment**: Test `math-python` environment - it uses Python tools for math problems (simpler than MuSiQue)

**Environment Details** (from `/tmp/verifiers/environments/math-python/`):
- **Type**: Tool-using math environment with Python execution
- **Complexity**: Multi-step but simpler than MuSiQue (fewer tools, clearer task)
- **Task**: Math problems requiring Python tool calls
- **Parser**: Basic boxed answer extraction
- **Reward**: Binary correctness via symbolic verification

**Setup**:
```bash
# Install math-python environment
vf-install math-python --from-repo

# Train with same parameters as working GSM8K
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_math_python.py train \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --batch-size 8 --num-generations 8 \
    --learning-rate 1e-6 \
    --num-epochs 3 \
    2>&1 | tee outputs/h07-math-python-$(date +%s).log
```

**Success Criteria**: 
- **If math-python shows improvement**: Issue is MuSiQue-specific (reward function, retrieval complexity, etc.)
- **If math-python also fails to improve**: Issue is general multi-step/tool-based environment problem

### Method 2: Compare Conversation Structures
**Analysis**: Compare conversation formats between environments:

**GSM8K Format**:
```
User: [question]
Assistant: [reasoning] → [answer]
```

**Math-Python Format**:
```
User: [question]  
Assistant: [reasoning] → [tool call: python]
Tool: [python execution result]
Assistant: [final answer]
```

**MuSiQue Format**:
```
User: [question]
Assistant: [tool call: retrieve]
Tool: [documents]
Assistant: [tool call: retrieve] 
Tool: [more documents]
Assistant: [reasoning with <think><cite><answer> tags]
```

### Method 3: Token Masking Analysis for Multi-step
**Hypothesis**: Multi-step environments have more complex masking requirements
- Tool response tokens should be masked out
- Only assistant reasoning tokens should contribute to loss
- Multi-turn structure may confuse masking logic

## Invalidation Criteria

This hypothesis is **invalidated** if:
1. **Math-python shows same learning pattern as GSM8K** (improvement)
2. **Other multi-step environments work fine**  
3. **Issue is definitively traced to MuSiQue-specific components**

If validated, this narrows the problem significantly to multi-step tool interaction patterns.

## Current Status: UNTESTED

## Next Steps Based on Validation

**If H07 is Validated (multi-step environments generally fail)**:
- Focus on token masking in multi-turn conversations (H02)
- Investigate conversation state management
- Check tool response handling in trainer

**If H07 is Invalidated (math-python works fine)**:
- Issue is MuSiQue-specific
- Focus on MuSiQue reward function, retrieval complexity, context length
- Prioritize H04 (context truncation) and H05 (reward signal issues)

## Experimental Priority

**Test Order**: 2 (high priority after H06 library validation)
**Reason**: 
- Critical to isolate multi-step vs MuSiQue-specific issues
- Relatively quick test (math-python is simpler than MuSiQue)
- Will guide all subsequent debugging efforts

**Time Estimate**: 2-3 hours for math-python installation and training test

---

**Strategic Importance**: This hypothesis is key to determining whether to focus on general multi-step environment issues vs MuSiQue-specific problems.

# math-python

Install environment

```sh
vf-install math-python --from-repo
```


Inference 

```sh
CUDA_VISIBLE_DEVICES=0 vf-vllm --model "Qwen/Qwen2.5-7B-Instruct" \
    --port 8000 \
    --gpu-memory-utilization 0.6 \
    --max-model-len 8192 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --enforce-eager
```

Training
```sh
CUDA_VISIBLE_DEVICES=1 python scripts/train_math_python.py 2>&1 | tee outputs/logs/train-$(date +%s).log
```

Training multi-GPU
```sh
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_math_python.py \
    2>&1 | tee outputs/logs/train-$(date +%s).log
```