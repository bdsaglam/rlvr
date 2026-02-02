# MuSiQue RL Training Debug: Problem Statement

## Overview

We are attempting to fine-tune large language models (7B parameters) using reinforcement learning on the MuSiQue multi-hop question answering dataset. Despite extensive hyperparameter tuning and multiple approaches, **the model performance does not improve during training** - rewards fluctuate around the baseline without showing an upward trend.

## Environment Setup

### Architecture
- **Model**: Primarily Qwen2.5-7B-Instruct, also tested Llama3.1-8B-Instruct, Qwen3-8B
- **Training**: GRPO (Group Relative Policy Optimization) via verifiers library
- **Environment**: Custom MuSiQue environment (`vf_musique`) with tool-based multi-hop reasoning
- **Infrastructure**: Multi-GPU training with DeepSpeed ZeRO Stage 3, LoRA fine-tuning

### Dataset & Task
- **Dataset**: MuSiQue multi-hop question answering
- **Format**: Questions requiring 1-2 hops across multiple documents
- **Tools**: Document retrieval (BM25, semantic, hybrid strategies)
- **Expected Output**: Models must use tools to retrieve documents, reason across them, and provide citations in structured format (`<think>`, `<cite>`, `<answer>` tags)

## Current Behavior

### What Works
- **Baseline Performance**: Pre-trained models achieve ~60-70% combined reward without fine-tuning
- **Successful Trajectories**: Models occasionally generate correct answers with proper format and citations
- **Environment Functionality**: Reward functions work correctly, successful completions get high rewards
- **Training Infrastructure**: No crashes, gradient flow appears normal

### The Problem
- **No Learning**: Reward scores fluctuate around baseline (60-70%) throughout training
- **No Upward Trend**: Despite seeing high-reward examples, average performance doesn't improve
- **Consistent Pattern**: This occurs across different models, hyperparameters, and configurations

## Baseline Performance Context

**Critical Detail**: The base model (without fine-tuning) already performs reasonably well:
- Gets correct answers ~60-70% of the time
- Occasionally uses proper citation format
- Can perform multi-hop reasoning to some degree
- This means we're trying to improve from a "decent" baseline, not from zero

## What We've Tried

### Models Tested
- Qwen/Qwen2.5-7B-Instruct (primary focus)
- meta-llama/Llama-3.1-8B-Instruct 
- willcb/Qwen3-8B
- willcb/Qwen3-4B

### Training Configurations
- **PEFT vs Full Fine-tuning**: Both LoRA (r=8,16,32,64, alpha=16,32,64,128) and full parameter updates
- **Loss Types**: `grpo`, `dr_grpo`, `bnpo`
- **Batch Sizes**: 8, 16 (per device)
- **Generation Counts**: 8, 16 completions per prompt
- **Learning Rates**: 1e-6, 1e-5, 5e-6
- **KL Penalties**: 0.0, 0.01, 0.04
- **Reward Scaling**: Both enabled and disabled
- **Gradient Clipping**: 0.01, 0.1
- **Sequence Lengths**: max_prompt=4096-8192, max_new_tokens=1024-2048

### Infrastructure Variations
- DeepSpeed ZeRO Stage 3 vs model replication
- Different GPU counts (2-3 processes)
- Various gradient accumulation steps (2,4,8)

### Dataset Variations  
- Full dataset vs mini dataset vs sliced datasets (train[:256])
- Different noise rates for non-supporting documents

## Key Observations

1. **Rewards Are Not Sparse**: Unlike typical RL scenarios, we get meaningful rewards regularly (~60-70% baseline)
2. **Format Parsing Works**: Models can generate the required XML tag format 
3. **Tools Function Correctly**: Document retrieval and environment responses work as expected
4. **No Error Patterns**: No systematic parsing failures, OOM errors, or gradient explosions visible
5. **GSM8K Comparison**: Currently testing if the same setup works on GSM8K dataset to isolate MuSiQue-specific vs library-wide issues

## Hypothesis Framework

Each hypothesis in this framework should address:
- **Description**: What might be wrong
- **Why It Explains the Behavior**: How it causes "no learning despite good baseline"
- **Validation Method**: Specific experiment or check to test it
- **Invalidation Criteria**: What result would rule it out
- **Priority**: High/Medium/Low based on likelihood and impact
- **Status**: Untested/Testing/Validated/Invalidated
- **Results**: What we found when tested

## Success Criteria

We'll consider this issue **resolved** when:
1. Training shows consistent upward trend in rewards over time
2. Evaluation on held-out set shows improvement over baseline
3. The improvement is reproducible across multiple runs
4. We can explain why the fix works

## Next Steps

Work through hypotheses systematically, starting with highest priority ones. Test one hypothesis at a time with minimal changes to isolate variables. Document all results thoroughly.

---

*This document serves as the master reference for debugging MuSiQue RL training issues. Update it as new information emerges.*