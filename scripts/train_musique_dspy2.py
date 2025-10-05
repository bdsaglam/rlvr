#!/usr/bin/env python
"""
Train a model on MuSiQue dataset using DSPy with Arbor RL server.

This script implements multi-hop question answering training using DSPy's GRPO
optimizer, adapting the DSPy GRPO approach for MuSiQue's multi-hop QA task.

Prerequisites:
0. Setup environment
    just setup

1. Start Arbor RL server
   python -m arbor.cli serve --arbor-config arbor.yaml

2. Start services
   docker-compose up
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import dspy
import torch
import typer
from dotenv import load_dotenv
from dspy.clients.lm_local_arbor import ArborProvider
from dspy.teleprompt.grpo import GRPO

from rlvr.dspy.mhqa.baleen import MultiHopQA
from rlvr.dspy.mhqa.data import prepare_musique_dataset
from rlvr.dspy.mhqa.metrics import metric

assert load_dotenv(), "Failed to load .env file"

app = typer.Typer()


def setup_mlflow():
    import mlflow
    import mlflow.dspy

    mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))
    mlflow.set_experiment("rlvr-dspy-musique")
    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True,
    )
    typer.echo(f"✅ MLflow tracking enabled at {os.getenv('MLFLOW_URL')}")


def get_model_name(model_path: str) -> str:
    """Extract model name from path."""
    if Path(model_path).exists():
        return Path(model_path).name
    else:
        return model_path.split("/")[-1]


@app.command()
def train(
    # Model configuration
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", "--model", "-m", help="Local model to train"),
    port: int = typer.Option(7453, "--port", help="Arbor server port"),
    temperature: float = typer.Option(0.7, "--temperature", help="Generation temperature"),
    # Dataset configuration
    datasets_str: str = typer.Option(
        "bdsaglam/musique,answerable,train", "--datasets", help="Datasets string in format 'name,subset,split'"
    ),
    eval_datasets_str: str = typer.Option(
        "bdsaglam/musique-mini,answerable,validation",
        "--eval-datasets",
        help="Evaluation datasets string in format 'name,subset,split'",
    ),
    noise_rate: float = typer.Option(1.0, "--noise-rate", help="Noise rate for filtering non-supporting documents"),
    # Training configuration
    num_train_steps: int = typer.Option(500, "--num-train-steps", help="Number of training steps"),
    num_examples_per_step: int = typer.Option(0, "--num-examples-per-step", help="Examples per GRPO step"),
    num_rollouts_per_step: int = typer.Option(8, "--num-rollouts-per-step", help="Rollouts per GRPO step"),
    batch_size: int = typer.Option(2, "--batch-size", help="Per-device batch size"),
    gradient_accumulation_steps: int = typer.Option(8, "--gradient-accumulation", help="Gradient accumulation steps"),
    learning_rate: float = typer.Option(2e-5, "--learning-rate", "-lr", help="Learning rate"),
    max_grad_norm: float = typer.Option(0.01, "--max-grad-norm", help="Maximum gradient norm"),
    kl_beta: float = typer.Option(0.00, "--kl-beta", help="KL divergence coefficient"),
    # LoRA configuration
    use_lora: bool = typer.Option(True, "--use-lora/--no-lora", help="Use LoRA for training"),
    # Output configuration
    output_dir: Path = typer.Option("./outputs", "-o", help="Output directory"),
    run_name: Optional[str] = typer.Option(None, help="Run name (auto-generated if not provided)"),
    # Experiment tracking
    use_mlflow: bool = typer.Option(True, "--mlflow/--no-mlflow", help="Use MLflow for tracking"),
):
    """Train a model on MuSiQue using DSPy GRPO for multi-hop question answering."""

    # Generate run name if not provided
    if run_name is None:
        model_name = get_model_name(model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}-musique-dspy-{timestamp}"

    # Create output directory
    output_dir = Path(output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"📁 Output directory: {output_dir}")

    # Print configuration
    typer.echo("🚀 Starting MuSiQue DSPy Training")
    typer.echo("=" * 50)
    typer.echo(f"📝 Model: {model}")
    typer.echo(f"🏷️  Run name: {run_name}")
    typer.echo(f"🌐 Arbor port: {port}")
    typer.echo(f"📊 Train dataset: {datasets_str}")
    typer.echo(f"🎲 Noise rate: {noise_rate}")
    typer.echo(f"🎓 Training: {num_train_steps} steps")
    typer.echo(f"📈 Learning rate: {learning_rate}")
    typer.echo(f"🎯 LoRA: {'enabled' if use_lora else 'disabled'}")
    typer.echo("=" * 50)

    # Setup MLflow if requested
    if use_mlflow:
        try:
            setup_mlflow()
        except ImportError:
            typer.echo("⚠️  MLflow not installed, continuing without tracking")
            use_mlflow = False

    # Setup DSPy with Arbor provider
    typer.echo("🤖 Setting up DSPy with Arbor...")
    local_lm = dspy.LM(
        model=f"openai/arbor:{model}",
        provider=ArborProvider(),
        temperature=temperature,
        api_base=f"http://localhost:{port}/v1/",
        api_key="arbor",
    )
    dspy.configure(lm=local_lm)

    # Load MuSiQue dataset
    typer.echo("📊 Loading MuSiQue training dataset...")
    trainset = prepare_musique_dataset(datasets_str=datasets_str, noise_rate=noise_rate)

    # Load separate eval dataset if provided, otherwise use train data
    typer.echo("📊 Loading MuSiQue evaluation dataset...")
    devset = prepare_musique_dataset(datasets_str=eval_datasets_str, noise_rate=noise_rate)

    typer.echo(f"✅ Dataset loaded: {len(trainset)} train, {len(devset)} dev")

    # Show example
    typer.echo("\n📝 Example question:")
    typer.echo(f"Q: {trainset[0].question}")
    typer.echo(f"A: {trainset[0].answer}")
    typer.echo(f"Supporting doc IDs: {trainset[0].supporting_ids}")
    typer.echo(f"Number of hops: {trainset[0].n_hops}")

    # Create the MultiHopQA program
    typer.echo("\n🔧 Creating MultiHopQA program...")
    program = MultiHopQA()
    program.set_lm(local_lm)

    # Setup GRPO training
    typer.echo("\n🎓 Setting up GRPO optimization...")

    train_kwargs = {
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "temperature": temperature,
        "beta": kl_beta,
        "learning_rate": learning_rate,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "bf16": True,
        "lr_scheduler_type": "constant_with_warmup",
        "max_prompt_length": None,
        "max_completion_length": None,
        "scale_rewards": True,
        "max_grad_norm": max_grad_norm,
        "lora": use_lora,
        "report_to": "wandb",
        "output_dir": str(output_dir),
    }

    compiler = GRPO(
        metric=metric,
        num_dspy_examples_per_grpo_step=num_examples_per_step,
        num_rollouts_per_grpo_step=num_rollouts_per_step,
        exclude_demos=True,
        num_train_steps=num_train_steps,
        num_threads=24,
        use_train_as_val=False,
        num_steps_for_val=10,
        train_kwargs=train_kwargs,
        report_train_scores=False,
    )

    # Train with GRPO
    typer.echo("\n🔥 Starting GRPO training!")
    typer.echo("=" * 50)

    try:
        optimized_program = compiler.compile(
            student=program,
            trainset=trainset,
            valset=devset,
        )
        typer.echo("\n✅ Training completed successfully!")

        # Save optimized program
        optimized_program.save(output_dir / "optimized_program.json")
        typer.echo(f"💾 Optimized program saved to: {output_dir / 'optimized_program.json'}")

        # Print next steps
        typer.echo("\n📝 Next steps:")
        typer.echo(f"   1. Find outputs in: {output_dir}")
        typer.echo("   2. The model checkpoint is managed by Arbor server")
        typer.echo("   3. Evaluate further with different metrics")

    except KeyboardInterrupt:
        typer.echo("\n⚠️ Training interrupted by user")
        typer.echo(f"💾 Partial results may be in: {output_dir}")

    except Exception as e:
        typer.echo(f"\n❌ Training failed with error: {e}")
        raise

    finally:
        # Cleanup
        if use_mlflow:
            try:
                import mlflow

                mlflow.end_run()
            except Exception:
                pass
        torch.cuda.empty_cache()


@app.command()
def evaluate(
    model: str = typer.Option("Qwen/Qwen2.5-7B-Instruct", "--model", "-m", help="Model to evaluate"),
    port: int = typer.Option(7453, "--port", help="Arbor server port"),
    datasets_str: str = typer.Option(
        "bdsaglam/musique-mini,answerable,validation",
        "--datasets",
        help="Datasets string in format 'name,subset,split'",
    ),
    output_file: Path = typer.Option("./outputs/dspy-musique-evaluation-results.json", "-o", help="Output file"),
):
    """Evaluate a trained model on MuSiQue test set."""

    typer.echo("🔮 Starting MuSiQue DSPy evaluation")
    typer.echo("=" * 50)
    typer.echo(f"📝 Model: {model}")
    typer.echo(f"📊 Dataset: {datasets_str}")
    typer.echo("=" * 50)

    # Setup DSPy
    local_lm = dspy.LM(
        model=f"openai/arbor:{model}",
        provider=ArborProvider(),
        temperature=0.5,  # Lower temperature for evaluation
        api_base=f"http://localhost:{port}/v1/",
        api_key="arbor",
    )
    dspy.configure(lm=local_lm)

    # Load test dataset
    typer.echo("📊 Loading test dataset...")
    dataset = prepare_musique_dataset(datasets_str=datasets_str, noise_rate=1.0)
    typer.echo(f"✅ Loaded {len(dataset)} test examples")

    # Create program
    program = MultiHopQA()
    program.set_lm(local_lm)

    # Evaluate
    evaluate = dspy.Evaluate(
        devset=dataset,
        metric=metric,
        num_threads=16,
        display_progress=True,
        display_table=10,
    )

    result = evaluate(program)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    typer.echo(f"\n✅ Evaluation results saved to: {output_file}")


if __name__ == "__main__":
    app()
