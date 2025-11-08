from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import typer
import verifiers as vf
import wandb
from accelerate import Accelerator
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

try:
    from unsloth import FastLanguageModel
except ImportError as e:
    raise ImportError("Unsloth is required for this training script. Install with: pip install unsloth") from e

# Import enhanced trainer for better logging
from rlvr.trainers import EnhancedGRPOTrainer
from rlvr.utils.repro import set_seed

assert load_dotenv(), "Failed to load .env file"

accelerator = Accelerator()

app = typer.Typer()


set_seed(42)


def setup_obs(run_name: str):
    import mlflow

    # Tell MLflow about the server URI.
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Enable autologging with all features
    mlflow.openai.autolog()
    # Create a unique name for your experiment.
    mlflow.set_experiment(f"rlvr-train-{run_name}")


def get_model_name(model_path: str) -> str:
    """Extract model name from path."""
    if Path(model_path).exists():
        # For local paths, get the directory name
        return Path(model_path).name
    else:
        # For HuggingFace names, get the part after the slash
        return model_path.split("/")[-1]


@app.command()
def train(
    # Environment arguments
    env_id: str = typer.Option("vf-musique", "--env-id", "--env", help="Environment ID"),
    datasets_str: str = typer.Option(
        "bdsaglam/musique,answerable,train",
        "--datasets",
        help="Datasets string in format 'path,name,split;path2,name2,split2'",
    ),
    eval_datasets_str: str | None = typer.Option(
        None,
        "--eval-datasets",
        help="Datasets string in format 'path,name,split;path2,name2,split2'",
    ),
    noise_rate: float = typer.Option(1.0, "--noise", help="Noise rate to use for filtering non-supporting documents"),
    retriever: str = typer.Option("hybrid", help="Retrieval strategy to use"),
    max_concurrent: int = typer.Option(16, help="Maximum concurrent requests"),
    # Model arguments
    model: str = typer.Option(
        "unsloth/Qwen2.5-7B-Instruct", "--model", "-m", help="Model path or HuggingFace model name"
    ),
    # Generation parameters
    max_prompt_length: int = typer.Option(8192, help="Maximum prompt length"),
    max_tokens: int = typer.Option(1024, help="Maximum new tokens to generate"),
    temperature: float = typer.Option(0.5, help="Generation temperature"),
    top_p: float = typer.Option(0.95, help="Top-p sampling"),
    min_p: Optional[float] = typer.Option(None, help="Minimum probability for min-p sampling"),
    # Training arguments
    num_epochs: int = typer.Option(1, help="Number of training epochs"),
    max_steps: int = typer.Option(-1, help="Maximum number of training steps"),
    batch_size: int = typer.Option(8, help="Per-device batch size"),
    num_generations: int = typer.Option(8, help="Number of generations per prompt"),
    gradient_accumulation_steps: int = typer.Option(8, help="Gradient accumulation steps"),
    dtype: str = typer.Option("float32", help="Float data type. One of bfloat16, float16, float32"),
    # RL training parameters
    kl_beta: float = typer.Option(0.00, "--kl-beta", "--beta", help="KL divergence coefficient"),
    scale_rewards: bool = typer.Option(
        False, help="Scale rewards by group standard deviation during training. Original GRPO paper have this."
    ),
    loss_type: str = typer.Option("dr_grpo", help="Loss type"),
    num_iterations: int = typer.Option(1, help="Number of iterations per global batch (on-policy + off-policy)"),
    # LoRA arguments
    peft: bool = typer.Option(True, help="Use PEFT"),
    lora_r: int = typer.Option(16, help="LoRA rank (recommended: 8, 16, 32, 64, 128)"),
    lora_alpha: int = typer.Option(32, help="LoRA alpha (recommended: 1x or 2x of rank)"),
    lora_dropout: float = typer.Option(0.00, help="LoRA dropout"),
    # Optimizer arguments
    learning_rate: float = typer.Option(
        5e-6, "--learning-rate", "-lr", help="Learning rate (RL: 5e-6, Normal LoRA: 2e-4)"
    ),
    lr_scheduler_type: str = typer.Option(
        "cosine", help="Learning rate scheduler type (recommended: cosine or linear)"
    ),
    warmup_steps: int = typer.Option(10, help="Number of warmup steps (recommended: 5-10% of total steps)"),
    max_grad_norm: float = typer.Option(0.1, help="Maximum gradient norm for clipping"),
    weight_decay: float = typer.Option(0.01, help="Weight decay for regularization (recommended: 0.01-0.1)"),
    gradient_checkpointing: bool = typer.Option(True, help="Use gradient checkpointing"),
    # Evaluation arguments
    per_device_eval_batch_size: Optional[int] = typer.Option(None, help="Per-device evaluation batch size"),
    # Output arguments
    output_dir: Path = typer.Option("./outputs", "-o", help="Output directory"),
    run_name: Optional[str] = typer.Option(None, help="Run name (auto-generated if not provided)"),
    push_to_hub: bool = typer.Option(False, help="Push model to HuggingFace Hub"),
    hub_model_id: Optional[str] = typer.Option(None, help="Hub model ID (defaults to run_name)"),
    # WandB arguments
    report_to: str = typer.Option("wandb", help="Logging service (wandb, tensorboard, none)"),
    # Resume training
    resume_from_checkpoint: Optional[str] = typer.Option(None, help="Resume training from checkpoint"),
):
    """Train a model on MuSiQue using GRPO for multi-hop question answering."""

    # Generate run name if not provided
    if run_name is None:
        model_name = get_model_name(model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}-musique-rlvr-{timestamp}"

    # Set hub_model_id if not provided but push_to_hub is True
    if hub_model_id is None and push_to_hub:
        hub_model_id = run_name

    # Print training configuration
    typer.echo("🚀 Starting MuSiQue training")
    typer.echo("=" * 50)
    typer.echo(f"📝 Model: {model}")
    typer.echo(f"🏷️  Run name: {run_name}")
    typer.echo(f"📊 Datasets: {datasets_str}")
    typer.echo(f"📊 Evaluation datasets: {eval_datasets_str}")
    typer.echo(f"🔍 Retriever: {retriever}")
    typer.echo(f"🎲 Noise rate: {noise_rate}")
    typer.echo(f"📏 Batch size: {batch_size}")
    typer.echo(f"🔢 Generations: {num_generations}")
    typer.echo(f"📈 Learning rate: {learning_rate}")
    typer.echo(f"📊 LR Scheduler: {lr_scheduler_type}")
    typer.echo(f"🔥 Weight decay: {weight_decay}")
    typer.echo(f"🔄 Epochs: {num_epochs}")
    typer.echo(f"🔢 Max steps: {max_steps}")
    typer.echo(f"🎯 PEFT: {'enabled' if peft else 'disabled'}")
    if peft:
        typer.echo(f"   - Rank: {lora_r}, Alpha: {lora_alpha}, Dropout: {lora_dropout}")
        effective_batch_size = batch_size * gradient_accumulation_steps
        typer.echo(
            f"📦 Effective batch size: {effective_batch_size} (batch_size={batch_size} × grad_accum={gradient_accumulation_steps})"
        )
    typer.echo("=" * 50)

    # Load MuSiQue environment
    typer.echo("🌍 Loading MuSiQue environment...")
    vf_env = vf.load_environment(
        env_id=env_id,
        datasets_str=datasets_str,
        eval_datasets_str=eval_datasets_str,
        noise_rate=noise_rate,
        retriever=retriever,
    )
    typer.echo(f"✅ Environment loaded with {len(vf_env.dataset)} training examples")

    # if accelerator.is_main_process:
    # setup_obs(run_name=run_name)

    # Load model and tokenizer using Unsloth
    typer.echo(f"🤖 Loading model with Unsloth: {model}")

    # Use Unsloth's FastLanguageModel for optimized loading
    max_seq_length = max_prompt_length + max_tokens + 128
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model,
        max_seq_length=max_seq_length,
        load_in_4bit=False,  # Use False for LoRA 16bit training
        load_in_16bit=True,
        max_lora_rank=lora_r if peft else 0,
        fast_inference=False,  # Enable vLLM-compatible fast inference
    )
    typer.echo("✅ Model and tokenizer loaded with Unsloth")

    # Create training configuration
    typer.echo("⚙️ Setting up training configuration...")
    training_args = vf.grpo_defaults(run_name=run_name)

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override with custom arguments
    training_args.push_to_hub = push_to_hub
    training_args.report_to = report_to
    training_args.run_name = run_name
    training_args.output_dir = output_dir / run_name
    training_args.save_steps = 100
    training_args.save_strategy = "steps"
    training_args.save_only_model = False

    training_args.logging_steps = 1
    training_args.log_completions = True
    training_args.num_completions_to_print = 5  # Sample size to log
    training_args.num_train_epochs = num_epochs
    training_args.max_steps = max_steps
    training_args.shuffle_dataset = False
    training_args.max_prompt_length = max_prompt_length
    training_args.max_tokens = max_tokens
    training_args.max_seq_len = max_seq_length
    training_args.temperature = temperature
    training_args.min_p = min_p
    training_args.top_p = top_p
    training_args.top_k = None
    training_args.per_device_train_batch_size = batch_size
    training_args.num_generations = num_generations
    training_args.gradient_accumulation_steps = gradient_accumulation_steps
    training_args.max_grad_norm = max_grad_norm
    # Disable gradient checkpointing in training_args since Unsloth handles it
    training_args.gradient_checkpointing = False
    # training_args.gradient_checkpointing_kwargs = {
    #     "use_reentrant": False,
    # }
    training_args.learning_rate = learning_rate
    training_args.lr_scheduler_type = lr_scheduler_type
    training_args.warmup_steps = warmup_steps
    training_args.weight_decay = weight_decay
    training_args.adam_beta1 = 0.9
    training_args.adam_beta2 = 0.99
    if dtype == "bfloat16":
        training_args.bf16 = True
        training_args.fp16 = False
    elif dtype == "float16":
        training_args.bf16 = False
        training_args.fp16 = True
    elif dtype == "float32":
        training_args.bf16 = False
        training_args.fp16 = False
    else:
        raise ValueError(f"Invalid float precision: {dtype}")

    training_args.beta = kl_beta
    training_args.loss_type = loss_type
    training_args.num_iterations = num_iterations
    training_args.scale_rewards = scale_rewards

    training_args.max_concurrent = max_concurrent
    training_args.async_generation_timeout = 1200

    if eval_datasets_str:
        training_args.eval_strategy = "steps"
        training_args.eval_steps = 100
        training_args.per_device_eval_batch_size = (
            per_device_eval_batch_size if per_device_eval_batch_size is not None else batch_size * 2
        )
        training_args.eval_on_start = True
    else:
        training_args.eval_strategy = "no"
        training_args.eval_on_start = False

    if push_to_hub:
        training_args.hub_model_id = hub_model_id

    # Set up LoRA config and apply with Unsloth
    if peft:
        typer.echo(f"🎯 Applying LoRA with Unsloth: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

        # Apply PEFT using Unsloth's optimized method
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            use_gradient_checkpointing="unsloth",  # Use Unsloth's optimized checkpointing
            random_state=89,
        )
        typer.echo("✅ LoRA applied with Unsloth optimizations")

    # Create trainer - PEFT already applied by Unsloth, so pass None
    typer.echo("🏃 Creating Enhanced GRPO trainer with full trajectory logging...")
    trainer = EnhancedGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        peft_config=None,  # Already applied by Unsloth above
        sync_ref_model=False,
    )
    typer.echo("✅ Enhanced trainer created with Unsloth-optimized model")

    # Print final configuration
    typer.echo("\n📋 Final Training Configuration:")
    typer.echo(f"📁 Output directory: {training_args.output_dir}")
    typer.echo(f"🚀 Push to hub: {'Yes' if push_to_hub else 'No'}")
    typer.echo(f"📝 Report to: {report_to}")

    # Start training
    typer.echo("\n🔥 Starting training!")
    typer.echo("=" * 50)

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        typer.echo("\n✅ Training completed successfully!")

        # Update experiment configs
        if wandb.run is not None and accelerator.is_main_process:
            wandb.run.config.update(
                {
                    "datasets": datasets_str,
                    "noise_rate": noise_rate,
                    "retriever": retriever,
                    "max_prompt_length": max_prompt_length,
                    "max_tokens": max_tokens,
                    "num_generations": num_generations,
                    "temperature": temperature,
                    "num_epochs": num_epochs,
                    "max_steps": max_steps,
                    "batch_size": batch_size,
                    "scale_rewards": scale_rewards,
                    "loss_type": loss_type,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "learning_rate": learning_rate,
                    "kl_beta": kl_beta,
                    "peft": peft,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                }
            )

        # Print next steps
        typer.echo("\n📝 Next steps:")
        typer.echo(f"   1. Find your model in: {training_args.output_dir}")
        typer.echo(f"   2. Evaluate with: python scripts/evaluate_musique.py --model {training_args.output_dir}")
        if push_to_hub:
            typer.echo(f"   3. Check HuggingFace Hub: https://huggingface.co/{run_name}")

    except KeyboardInterrupt:
        typer.echo("\n⚠️ Training interrupted by user")
        typer.echo(f"💾 Checkpoints saved in: {training_args.output_dir}")
        typer.echo(f"🔄 Resume with: --resume-from-checkpoint {training_args.output_dir}")

    except Exception as e:
        typer.echo(f"\n❌ Training failed with error: {e}")
        typer.echo(f"💾 Check logs and checkpoints in: {training_args.output_dir}")
        raise

    finally:
        # Cleanup
        if accelerator.is_main_process:
            wandb.finish()
        del model
        del trainer
        torch.cuda.empty_cache()


@app.command()
def evaluate(
    env_id: str = typer.Option("vf-musique", "--env-id", "--env", help="Environment ID"),
    datasets_str: str = typer.Option(
        "bdsaglam/musique-mini,answerable,validation",
        "--datasets",
        help="Datasets string in format 'path,name,split;path2,name2,split2'",
    ),
    noise_rate: float = typer.Option(1.0, "--noise", help="Noise rate to use for filtering non-supporting documents"),
    retriever: str = typer.Option("hybrid", "--retriever", help="Retrieval strategy"),
    max_concurrent: int = typer.Option(16, help="Maximum concurrent requests"),
    model: str = typer.Option("unsloth/Qwen2.5-7B-Instruct", "--model", "-m", help="Model to use for evaluation"),
    temperature: float = typer.Option(0.5, "--temperature", help="Generation temperature"),
    max_tokens: int = typer.Option(1024, "--max-new-tokens", help="Maximum new tokens to generate"),
    output_file: Path = typer.Option("./outputs/evaluation-results.jsonl", "-o"),
) -> Dataset:
    """Evaluate a model on MuSiQue dataset."""

    typer.echo("🔮 Starting MuSiQue evaluation")
    typer.echo("=" * 50)
    typer.echo(f"📝 Model: {model}")
    typer.echo(f"📊 Dataset: {datasets_str} (noise rate: {noise_rate})")
    typer.echo(f"🔍 Retriever: {retriever}")
    typer.echo(f"🌡️ Temperature: {temperature}")
    typer.echo(f"🎯 Max new tokens: {max_tokens}")
    typer.echo(f"💾 Output: {output_file}")
    typer.echo("=" * 50)

    # Load MuSiQue environment
    typer.echo("🌍 Loading MuSiQue environment...")

    vf_env = vf.load_environment(
        env_id=env_id,
        datasets_str=datasets_str,
        eval_datasets_str=datasets_str,
        noise_rate=noise_rate,
        retriever=retriever,
    )
    typer.echo(f"✅ Environment loaded with {len(vf_env.dataset)} examples")

    # Use OpenAI-compatible API client (e.g., for vLLM)
    typer.echo("🤖 Using OpenAI-compatible API client...")
    client = OpenAI()

    # Run evaluation using the environment
    typer.echo("🔄 Running evaluation...")
    result = vf_env.evaluate(
        client,
        model,
        rollouts_per_example=1,
        max_concurrent=max_concurrent,
        sampling_args={
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": None,
        },
    )
    result_dataset = vf_env.make_dataset(result)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_dataset.to_json(output_path, orient="records", lines=True)
    typer.echo(f"💾 Results saved to: {output_path}")

    return result_dataset


if __name__ == "__main__":
    app()
