# Enhanced W&B Logging Implementation

## Summary

We've successfully implemented an enhanced GRPO trainer with comprehensive W&B logging that captures full conversation trajectories, including all tool interactions. This addresses the critical debugging issue where the original verifiers library only logs the first and last messages, making it impossible to debug multi-step tool-based environments.

## What Was Implemented

### 1. **EnhancedGRPOTrainer** (`src/rlvr/trainers/enhanced_grpo_trainer.py`)
- Subclasses the original `GRPOTrainer` from verifiers library
- Adds full trajectory logging to W&B with comprehensive formatting
- Maintains compatibility with existing training scripts
- Key features:
  - Logs complete multi-turn conversations
  - Captures all tool calls and responses
  - Shows reward breakdowns per component
  - Tracks trajectory statistics (turns, tool usage, etc.)

### 2. **Logging Helper Utilities** (`src/rlvr/utils/logging_helpers.py`)
- `format_conversation()`: Formats multi-turn conversations with visual indicators
- `extract_trajectory_stats()`: Extracts statistics like turn counts, tool usage
- `format_input_data()`: Formats input data for logging
- `format_reward_components()`: Creates readable reward breakdowns

### 3. **Updated Training Scripts**
- **`scripts/train_musique.py`**: Now uses EnhancedGRPOTrainer with configurable logging options
- **`scripts/train_math_python.py`**: Updated to use enhanced logging
- Added command-line flags:
  - `--log-full-trajectories`: Enable/disable full trajectory logging (default: True)
  - `--log-trajectory-samples`: Number of samples to log per step (default: 5)
  - `--log-token-details`: Enable token-level debugging (default: False)

## How to Use

### Basic Usage

Simply run your training as before - the enhanced logging is enabled by default:

```bash
# MuSiQue training with enhanced logging
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch \
    --num-processes 3 \
    --config-file configs/zero3.yaml \
    scripts/train_musique.py train \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --datasets "bdsaglam/musique-mini,answerable,train"
```

### Customizing Logging

Control logging behavior with command-line flags:

```bash
# Log more trajectory samples
scripts/train_musique.py train \
    --log-trajectory-samples 10 \
    --log-full-trajectories \
    --logging-steps 1
```

### Viewing Logs in W&B

The enhanced logging creates a new table called `full_trajectories` in W&B with columns:
- **Step**: Training step number
- **Sample**: Sample index within the step
- **Input**: Formatted input data (question, context, etc.)
- **Prompt**: Initial prompt/conversation
- **Full_Trajectory**: Complete conversation including all tool interactions
- **Rewards**: Formatted reward breakdown
- **Total_Reward**: Combined reward value
- **Trajectory Statistics**: Turns, tool calls, tool responses, etc.
- **Individual Reward Components**: Separate columns for each reward type

## Benefits

1. **Complete Visibility**: See entire multi-turn conversations, not just first/last messages
2. **Tool Interaction Debugging**: Identify issues with tool calls and responses
3. **Reward Attribution**: Understand which reward components are working/failing
4. **Learning Progress**: Track how model's tool usage evolves during training
5. **Format Verification**: Ensure proper XML tag usage (<think>, <cite>, <answer>)

## Example Output

In W&B, you'll see formatted trajectories like:

```
👤 USER:
What is the capital of France?

🤖 ASSISTANT:
I'll search for information about France's capital.

📞 TOOL CALLS:
  • retrieve({"query": "capital of France"})

🔧 TOOL [retrieve]:
Document ID: 1
Paris is the capital city of France...

🤖 ASSISTANT:
<think>
I found that Paris is the capital of France.
</think>
<cite>1</cite>
<answer>Paris</answer>
```

## Testing

Verify the implementation works:

```bash
# Test logging helpers
python scripts/test_logging_helpers.py

# Test with small training run
python scripts/test_enhanced_logging.py  # Note: requires vLLM server
```

## Key Files

- **Implementation**:
  - `src/rlvr/trainers/enhanced_grpo_trainer.py` - Enhanced trainer
  - `src/rlvr/utils/logging_helpers.py` - Helper functions
  
- **Updated Scripts**:
  - `scripts/train_musique.py` - MuSiQue training with enhanced logging
  - `scripts/train_math_python.py` - Math-Python training with enhanced logging
  
- **Tests**:
  - `scripts/test_logging_helpers.py` - Test helper functions
  - `scripts/test_enhanced_logging.py` - Test full trainer

## Next Steps

With enhanced logging in place, you can now:

1. **Debug the eval/training mismatch** in math-python (22% eval vs 0% training)
2. **Analyze MuSiQue trajectories** to identify why rewards don't improve
3. **Compare successful vs failed trajectories** to understand patterns
4. **Track tool usage evolution** during training

The enhanced logging should reveal exactly what's happening during training and help identify the root cause of the learning issues.