import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import typer
import verifiers as vf
from accelerate import Accelerator
from datasets import Dataset
from dotenv import load_dotenv
from openai import OpenAI

import wandb

assert load_dotenv(), "Failed to load .env file"

accelerator = Accelerator()

app = typer.Typer()


def setup_obs(run_name: str):
    import mlflow

    # Tell MLflow about the server URI.
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Enable autologging with all features
    mlflow.openai.autolog()
    # Create a unique name for your experiment.
    mlflow.set_experiment(run_name)


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
    # Model arguments
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", "--model", "-m", help="Model path or HuggingFace model name"),
    # Dataset arguments
    datasets_str: str = typer.Option(
        "bdsaglam/musique,answerable,train",
        "--datasets",
        help="Datasets string in format 'path,name,split;path2,name2,split2'",
    ),
    noise_rate: float = typer.Option(
        1.0, "--noise-rate", help="Noise rate to use for filtering non-supporting documents"
    ),
    # Retriever arguments
    retriever: str = typer.Option("hybrid", "--retriever", help="Retrieval strategy to use"),
    # Environment arguments
    max_prompt_length: int = typer.Option(4096, "--max-prompt-length", help="Maximum prompt length"),
    max_completion_length: int = typer.Option(1024, "--max-completion-length", help="Maximum completion length"),
    # Training arguments
    batch_size: int = typer.Option(8, "--batch-size", help="Per-device batch size"),
    num_generations: int = typer.Option(8, "--num-generations", help="Number of generations per prompt"),
    gradient_accumulation_steps: int = typer.Option(
        8, "--gradient-accumulation-steps", help="Gradient accumulation steps"
    ),
    learning_rate: float = typer.Option(1e-5, "--learning-rate", help="Learning rate"),
    num_epochs: int = typer.Option(1, "--num-epochs", help="Number of training epochs"),
    max_steps: int = typer.Option(500, "--max-steps", help="Maximum training steps"),
    save_steps: int = typer.Option(100, "--save-steps", help="Save checkpoint every N steps"),
    eval_steps: int = typer.Option(50, "--eval-steps", help="Evaluate every N steps"),
    # Additional training parameters
    temperature: float = typer.Option(1.0, "--temperature", help="Generation temperature"),
    kl_beta: float = typer.Option(0.1, "--kl-beta", help="KL divergence coefficient"),
    scale_rewards: bool = typer.Option(False, "--scale-rewards", help="Scale rewards during training"),
    loss_type: str = typer.Option("dr_grpo", "--loss-type", help="Loss type"),
    num_iterations: int = typer.Option(
        1, "--num-iterations", help="Number of iterations per global batch (on-policy + off-policy)"
    ),
    # LoRA arguments
    peft: bool = typer.Option(True, "--peft/--no-peft", help="Use PEFT"),
    lora_r: int = typer.Option(32, "--lora-r", help="LoRA rank"),
    lora_alpha: int = typer.Option(64, "--lora-alpha", help="LoRA alpha"),
    lora_dropout: float = typer.Option(0.05, "--lora-dropout", help="LoRA dropout"),
    # Optimizer arguments
    lr_scheduler_type: str = typer.Option(
        "constant_with_warmup", "--lr-scheduler-type", help="Learning rate scheduler type"
    ),
    warmup_steps: int = typer.Option(50, "--warmup-steps", help="Number of warmup steps"),
    adam_beta1: float = typer.Option(0.9, "--adam-beta1", help="Adam beta1 parameter"),
    adam_beta2: float = typer.Option(0.99, "--adam-beta2", help="Adam beta2 parameter"),
    max_grad_norm: float = typer.Option(0.5, "--max-grad-norm", help="Maximum gradient norm for clipping"),
    # Logging arguments
    logging_steps: int = typer.Option(1, "--logging-steps", help="Log every N steps"),
    log_completions: bool = typer.Option(
        True, "--log-completions/--no-log-completions", help="Log completions to wandb"
    ),
    log_on_each_node: bool = typer.Option(False, "--log-on-each-node/--no-log-on-each-node", help="Log on each node"),
    # Evaluation arguments
    eval_on_start: bool = typer.Option(
        False, "--eval-on-start/--no-eval-on-start", help="Run evaluation before training"
    ),
    per_device_eval_batch_size: Optional[int] = typer.Option(
        None, "--per-device-eval-batch-size", help="Per-device evaluation batch size"
    ),
    eval_accumulation_steps: int = typer.Option(
        1, "--eval-accumulation-steps", help="Number of evaluation accumulation steps"
    ),
    # Checkpointing arguments
    save_only_model: bool = typer.Option(
        False, "--save-only-model/--save-full-checkpoint", help="Save only model weights, not full checkpoint"
    ),
    # Output arguments
    output_dir: Path = typer.Option("./outputs", "--output-dir", "-o", help="Output directory"),
    run_name: Optional[str] = typer.Option(None, "--run-name", help="Run name (auto-generated if not provided)"),
    push_to_hub: bool = typer.Option(False, "--push-to-hub", help="Push model to HuggingFace Hub"),
    hub_model_id: Optional[str] = typer.Option(None, "--hub-model-id", help="Hub model ID (defaults to run_name)"),
    # WandB arguments
    report_to: str = typer.Option("wandb", "--report-to", help="Logging service (wandb, tensorboard, none)"),
    # Resume training
    resume_from_checkpoint: Optional[str] = typer.Option(
        None, "--resume-from-checkpoint", help="Resume training from checkpoint"
    ),
    # Mixed precision
    bf16: bool = typer.Option(True, "--bf16/--no-bf16", help="Use bfloat16 mixed precision"),
    # Gradient checkpointing
    gradient_checkpointing: bool = typer.Option(
        True, "--gradient-checkpointing/--no-gradient-checkpointing", help="Use gradient checkpointing"
    ),
):
    """Train a model on MuSiQue using GRPO for multi-hop question answering."""

    # Generate run name if not provided
    if run_name is None:
        model_name = get_model_name(model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}-musique-grpo-{timestamp}"

    # Set hub_model_id if not provided but push_to_hub is True
    if hub_model_id is None and push_to_hub:
        hub_model_id = run_name

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print training configuration
    typer.echo("🚀 Starting MuSiQue training")
    typer.echo("=" * 50)
    typer.echo(f"📝 Model: {model}")
    typer.echo(f"🏷️  Run name: {run_name}")
    typer.echo(f"📊 Datasets: {datasets_str}")
    typer.echo(f"🔍 Retriever: {retriever}")
    typer.echo(f"🎲 Noise rate: {noise_rate}")
    typer.echo(f"📏 Batch size: {batch_size}")
    typer.echo(f"🔢 Generations: {num_generations}")
    typer.echo(f"📈 Learning rate: {learning_rate}")
    typer.echo(f"🔄 Epochs: {num_epochs}")
    typer.echo(f"🎯 PEFT: {'enabled' if peft else 'disabled'}")
    if peft:
        typer.echo(f"   - Rank: {lora_r}, Alpha: {lora_alpha}, Dropout: {lora_dropout}")
    typer.echo("=" * 50)

    # Load MuSiQue environment
    typer.echo("🌍 Loading MuSiQue environment...")
    vf_env = vf.load_environment(
        env_id="vf-musique",
        datasets_str=datasets_str,
        noise_rate=noise_rate,
        retriever_name=retriever,
    )
    typer.echo(f"✅ Environment loaded with {len(vf_env.dataset)} training examples")

    if accelerator.is_main_process:
        setup_obs(run_name=run_name)

    # Load model and tokenizer
    typer.echo(f"🤖 Loading model: {model}")
    model, tokenizer = vf.get_model_and_tokenizer(model)
    typer.echo("✅ Model and tokenizer loaded")

    # Create training configuration
    typer.echo("⚙️ Setting up training configuration...")
    training_args = vf.grpo_defaults(run_name=run_name)

    # Override with custom arguments
    training_args.output_dir = output_dir / run_name
    training_args.per_device_train_batch_size = batch_size
    training_args.num_generations = num_generations
    training_args.gradient_accumulation_steps = gradient_accumulation_steps
    training_args.learning_rate = learning_rate
    training_args.num_train_epochs = num_epochs
    training_args.max_steps = max_steps
    training_args.save_steps = save_steps
    training_args.eval_steps = eval_steps
    training_args.eval_strategy = "steps" if eval_steps > 0 else "no"
    training_args.save_strategy = "steps"
    training_args.push_to_hub = push_to_hub
    training_args.report_to = report_to
    training_args.run_name = run_name
    training_args.temperature = temperature
    training_args.beta = kl_beta
    training_args.max_prompt_length = max_prompt_length
    training_args.max_completion_length = max_completion_length
    training_args.num_iterations = num_iterations
    training_args.lr_scheduler_type = lr_scheduler_type
    training_args.warmup_steps = warmup_steps
    training_args.adam_beta1 = adam_beta1
    training_args.adam_beta2 = adam_beta2
    training_args.max_grad_norm = max_grad_norm
    training_args.logging_steps = logging_steps
    training_args.log_completions = log_completions
    training_args.log_on_each_node = log_on_each_node
    training_args.eval_on_start = eval_on_start
    training_args.eval_accumulation_steps = eval_accumulation_steps
    training_args.save_only_model = save_only_model
    training_args.bf16 = bf16
    training_args.gradient_checkpointing = gradient_checkpointing
    training_args.loss_type = loss_type
    training_args.scale_rewards = scale_rewards

    # Set evaluation batch size (default to training batch size if not provided)
    if per_device_eval_batch_size is not None:
        training_args.per_device_eval_batch_size = per_device_eval_batch_size
    else:
        training_args.per_device_eval_batch_size = batch_size

    if push_to_hub:
        training_args.hub_model_id = hub_model_id

    # Set up LoRA config
    lora_config = None
    if peft:
        lora_config = vf.lora_defaults(r=lora_r, alpha=lora_alpha)
        lora_config.dropout = lora_dropout
        lora_config.target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ]
        typer.echo(f"🎯 LoRA configuration: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")

    # Create trainer
    typer.echo("🏃 Creating GRPO trainer...")
    trainer = vf.GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        env=vf_env,
        args=training_args,
        peft_config=lora_config,
    )
    typer.echo("✅ Trainer created")

    # Print final configuration
    typer.echo("\n📋 Final Training Configuration:")
    typer.echo(f"📁 Output directory: {training_args.output_dir}")
    typer.echo(f"💾 Save every {save_steps} steps")
    typer.echo(f"📊 Evaluate every {eval_steps} steps" if eval_steps > 0 else "📊 No evaluation during training")
    typer.echo(f"🚀 Push to hub: {'Yes' if push_to_hub else 'No'}")
    typer.echo(f"📝 Report to: {report_to}")

    # Start training
    typer.echo("\n🔥 Starting training!")
    typer.echo("=" * 50)

    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        typer.echo("\n✅ Training completed successfully!")

        # Update experiment configs
        if wandb.run is not None:
            wandb.run.config.update(
                {
                    "datasets": datasets_str,
                    "noise_rate": noise_rate,
                    "retriever": retriever,
                    "max_prompt_length": max_prompt_length,
                    "max_completion_length": max_completion_length,
                    "num_generations": num_generations,
                    "temperature": temperature,
                    "num_epochs": num_epochs,
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
        wandb.finish()
        del model
        del trainer
        torch.cuda.empty_cache()


@app.command()
def evaluate(
    model: str = typer.Option("Qwen/Qwen2.5-3B-Instruct", "--model", "-m", help="Model to use for evaluation"),
    datasets_str: str = typer.Option(
        "bdsaglam/musique-mini,answerable,validation",
        "--datasets",
        help="Datasets string in format 'path,name,split;path2,name2,split2'",
    ),
    noise_rate: float = typer.Option(
        1.0, "--noise-rate", help="Noise rate to use for filtering non-supporting documents"
    ),
    retriever: str = typer.Option("hybrid", "--retriever", help="Retrieval strategy"),
    temperature: float = typer.Option(1.0, "--temperature", help="Generation temperature"),
    max_new_tokens: int = typer.Option(1024, "--max-new-tokens", help="Maximum tokens to generate"),
    output_file: Path = typer.Option("./outputs/evaluation-results.jsonl", "-o"),
) -> Dataset:
    """Evaluate a model on MuSiQue dataset."""

    typer.echo("🔮 Starting MuSiQue evaluation")
    typer.echo("=" * 50)
    typer.echo(f"📝 Model: {model}")
    typer.echo(f"📊 Dataset: {datasets_str} (noise rate: {noise_rate})")
    typer.echo(f"🔍 Retriever: {retriever}")
    typer.echo(f"🌡️ Temperature: {temperature}")
    typer.echo(f"🎯 Max tokens: {max_new_tokens}")
    typer.echo(f"💾 Output: {output_file}")
    typer.echo("=" * 50)

    # Load MuSiQue environment
    typer.echo("🌍 Loading MuSiQue environment...")

    vf_env = vf.load_environment(
        env_id="vf-musique",
        datasets_str=datasets_str,
        noise_rate=noise_rate,
        retriever_name=retriever,
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
        sampling_args={"temperature": temperature, "max_tokens": max_new_tokens},
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
