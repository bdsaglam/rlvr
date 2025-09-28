import os

import verifiers as vf
from dotenv import load_dotenv

# Import enhanced trainer for better logging
from rlvr.trainers import EnhancedGRPOTrainer

assert load_dotenv(), "Failed to load .env file"
os.environ["WANDB_PROJECT"] = "rlvr-debug"

vf_env = vf.load_environment(env_id="math-python")

model_name = "Qwen/Qwen2.5-7B-Instruct"
model, tokenizer = vf.get_model_and_tokenizer(model_name)
run_name = "math-python_" + model_name.split("/")[-1].lower()

training_args = vf.grpo_defaults(run_name=run_name)
training_args.max_tokens = 2048
training_args.max_prompt_length = 4096
training_args.max_seq_len = 8192
training_args.max_steps = 200
training_args.per_device_train_batch_size = 8
training_args.num_generations = 8
training_args.gradient_accumulation_steps = 1
training_args.beta = 0.1
training_args.max_grad_norm = 0.1
training_args.learning_rate = 1e-6
training_args.temperature = 0.5
training_args.top_p = 0.95
training_args.top_k = 50
training_args.report_to = "wandb"
training_args.log_completions = True
lora_config = vf.lora_defaults()
lora_config.target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]

print("🏃 Creating Enhanced GRPO trainer with full trajectory logging...")
trainer = EnhancedGRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
    peft_config=lora_config,
)
print("✅ Enhanced trainer created with full trajectory logging")

trainer.train()
