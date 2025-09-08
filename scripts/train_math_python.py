import os

import verifiers as vf
from dotenv import load_dotenv

# Import enhanced trainer for better logging
from src.rlvr.trainers import EnhancedGRPOTrainer

assert load_dotenv(), "Failed to load .env file"
os.environ['WANDB_PROJECT'] = "rlvr-debug"

vf_env = vf.load_environment(env_id="math-python")

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-python_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.per_device_train_batch_size = 8
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 8
training_args.max_tokens = 2048
training_args.max_prompt_length = 4096
training_args.max_seq_len = 8192
training_args.max_steps = 200
training_args.mask_env_responses = True
training_args.max_grad_norm = 0.1
training_args.beta = 0.1
training_args.report_to = "wandb"
training_args.log_completions = True
training_args.temperature = 0.5
training_args.top_p = 0.95
training_args.top_k = 50

# Enhanced logging configuration
training_args.log_full_trajectories = True
training_args.log_trajectory_samples = 5
training_args.log_token_details = False

print("🏃 Creating Enhanced GRPO trainer with full trajectory logging...")
trainer = EnhancedGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    peft_config=vf.lora_defaults(),
)
print("✅ Enhanced trainer created with full trajectory logging")

trainer.train()
