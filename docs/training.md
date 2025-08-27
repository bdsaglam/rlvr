Based on the [verifiers training documentation](https://verifiers.readthedocs.io/en/latest/training.html), here are all the training tips organized by category:

## **Before Training Tips**

1. **Evaluate baseline performance**: If your model gets 0% reward after 10+ attempts, the task is too hard
2. **Check task difficulty**: If baseline is already 80%+, consider harder examples  
3. **Ensure reward diversity**: You want varied scores within each generation group

## **Stability vs Performance Trade-offs**

### **For More Aggressive Training** (higher risk of collapse):
- Set `beta = 0` (no KL penalty)
- Increase learning rate (2e-6 to 5e-6)
- Increase `num_iterations` (2-4)

### **For More Stable Training** (slower progress):
- Increase `num_generations` (32-64)
- Increase batch size via `gradient_accumulation_steps`
- Decrease `max_grad_norm` (0.001-0.005)
- Use larger models (14B+)
- Keep `num_iterations = 1` (stay on-policy)

## **Best Practices**

### **Likely Beneficial:**
- Learning rate warmup (10-20 steps minimum)
- Periodic reference model updates for 500+ step runs
- One-step off-policy training (`num_batches_ahead = 1`)

### **Context-Dependent:**
- High `beta` values (0.1+) - more conservative
- Overlong filtering - depends on task
- Tool response masking - useful for multi-turn

## **Key Insight**
The best way to improve training is ensuring appropriate task difficulty for your model - not too easy, not too hard.

## **Hyperparameter Tips**

### **Batch Configuration:**
- `num_generations`: Larger groups (16-32) increase reward diversity but use more memory
- `per_device_train_batch_size`: Limited by GPU memory after model weights
- `gradient_accumulation_steps`: Use to achieve larger effective batch sizes

### **Generation Strategy:**
- High temperature (0.8-1.0) increases diversity within groups
- Consider your model's context window when setting lengths
- Longer completions allow more complex reasoning but increase memory usage

### **Training Dynamics:**
- Start with default `learning_rate = 1e-6` for stability
- `num_iterations > 1` does multiple updates per batch (more off-policy)
- Lower `max_grad_norm` for more stable but slower training

### **KL Regularization:**
- `beta = 0` removes reference model (faster, less stable)
- `beta = 0.001` is conservative; some use 0.01-0.1
- Sync reference model periodically for long runs

## **Troubleshooting Tips**

### **OOM during generation:**
- Reduce `num_generations` or `per_device_train_batch_size`
- Use LoRA instead of full finetuning
- Check vLLM server has sufficient memory

### **Training instability:**
- Reduce learning rate
- Decrease `max_grad_norm`
- Increase `beta` for stronger KL regularization

### **Poor reward diversity:**
- Increase temperature
- Check if task difficulty matches model capability
- Ensure your rubric differentiates quality levels

## **Infrastructure Tips**
- Ensure `huggingface` and `wandb` logins are configured
- Set `OPENAI_API_KEY` (can be dummy for vLLM)
- Increase ulimit for high concurrency: `ulimit -n 4096`
- For NCCL issues: try `NCCL_P2P_DISABLE=1`

These tips provide a comprehensive guide for training language models with GRPO using the Verifiers library, covering everything from initial setup to advanced optimization strategies.