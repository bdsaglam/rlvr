# Hypothesis H08: Evaluation vs Training Environment Mismatch

## Priority: CRITICAL

## Description

The math-python environment shows a major inconsistency: the same model gets **22% correct answer reward during evaluation** but **0% correct answer reward during training**. This suggests the evaluation and training pipelines are using different data formats, parsers, or reward computation logic.

## Root Cause Analysis

**Potential Issues**:

### 1. **Different Data Processing**
- Evaluation uses different prompt format than training
- Tool call formatting differs between eval and train
- Input tokenization or preprocessing diverges

### 2. **Parser Configuration Mismatch** 
- Evaluation parser vs training parser have different settings
- Tool call parsing works in eval but fails in training
- Answer extraction logic differs between pipelines

### 3. **Reward Function Bugs**
- Training reward function has bugs that evaluation doesn't
- Different reward computation logic between train/eval paths
- Token masking affecting reward calculation during training

### 4. **Environment State Differences**
- Training environment has different tool configurations
- Evaluation vs training use different tool response formats
- State management differs between eval and train modes

## Why This Explains Our Behavior

1. **False Baseline**: We thought the model could do 22%, but training shows it actually can't perform the task as formatted
2. **No Learning Signal**: If rewards are always 0 during training, there's no gradient signal
3. **Environment Bug**: The core issue may be environment implementation, not GRPO trainer

## Diagnostic Methods

### Method 1: Compare Exact Inputs/Outputs
**Compare identical examples between eval and training**:

```python
# In training script, add logging for first few examples
def log_training_vs_eval_comparison():
    # Log the exact prompt format used in training
    # Log the exact completion format during training  
    # Log the parsed output and reward calculation
    # Compare with evaluation notebook results for same examples
```

### Method 2: Manual Reward Calculation
**Manually compute rewards for training examples**:

```python
# Take a training completion and manually run it through eval reward function
training_completion = "[completion from training logs]" 
eval_reward = vf_env.rubric.compute_reward(training_completion, example_info)
print(f"Manual reward calculation: {eval_reward}")
```

### Method 3: Tool Call Format Inspection
**Check if tool calls work in both modes**:

```python
# Compare tool call formats
print("=== EVALUATION FORMAT ===")
print(eval_completion)
print("\n=== TRAINING FORMAT ===") 
print(training_completion)

# Check if tool calls are being parsed correctly in training
```

### Method 4: Environment Configuration Audit
**Compare environment settings**:

```python
# Check if environment is configured identically
eval_env = vf.load_environment(env_id="math-python", num_eval_examples=100)
train_env = vf.load_environment(env_id="math-python")  # training config

# Compare:
# - Tools available
# - Parser configuration  
# - Reward function settings
# - System prompts
```

## Expected Findings

**If H08 is Validated**:
- **Root Cause**: Environment implementation bug, not GRPO trainer issue
- **Fix**: Align training and evaluation environments
- **Impact**: May explain MuSiQue issues too (same environment bugs)

**If Training Format Issues Found**:
- Tool calls not formatted correctly during training
- Parser fails to extract answers from training completions
- Reward function sees malformed inputs

## Critical Questions

1. **Are the exact same prompts used in eval vs training?**
2. **Do tool calls work identically in both modes?**  
3. **Is the parser configuration identical?**
4. **Are rewards computed by the same function?**
5. **Do training completions look similar to eval completions?**

## Immediate Actions

1. **Log training completions** for first 5 examples and compare format to evaluation
2. **Manually run training completions through eval reward function**  
3. **Check if math-python training script uses same environment as evaluation**
4. **Verify tool call formatting in training vs eval**

## Current Status: CRITICAL INVESTIGATION NEEDED

This finding is **more important than the original MuSiQue issue** - if environments have fundamental eval/train mismatches, it affects all our debugging efforts.

## Hypothesis Status: HIGHLY LIKELY

The 22% vs 0% discrepancy is too large to be a coincidence. This strongly suggests a systematic difference between evaluation and training pipelines.

---

**Next Action**: Immediately investigate the exact format differences between evaluation and training completions for math-python.