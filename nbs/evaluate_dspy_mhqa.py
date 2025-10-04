import datetime
import os
import random
from pathlib import Path
from typing import Optional

import dspy
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from rlvr.dspy.mhqa.baleen import MultiHopQA
from rlvr.dspy.mhqa.metrics import metric

load_dotenv()

console = Console()
app = typer.Typer(help="Evaluate DSPy Multi-Hop QA models on MuSiQue dataset")


def setup_mlflow(experiment_name: str = "dspy-gepa-musique-evals"):
    """Set up MLflow tracking with auto-logging enabled."""
    try:
        import mlflow
        import mlflow.dspy
    except ImportError:
        console.print("❌ MLflow not available. Skipping experiment tracking.")
        return None

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    experiment = mlflow.set_experiment(experiment_name)
    mlflow.dspy.autolog(
        log_compiles=True,
        log_evals=True,
        log_traces_from_compile=True,
    )
    console.print(f"✅ MLflow tracking enabled at {os.getenv('MLFLOW_TRACKING_URI')}")
    return experiment


def setup_language_models(
    student_model: str,
    student_temperature: float,
    student_max_tokens: int,
    student_api_base: str,
    reflection_model: str,
    reflection_max_tokens: int,
    use_thinking: bool,
):
    """Configure the student and reflection language models."""
    # Student LM
    lm = dspy.LM(
        student_model,
        temperature=student_temperature,
        max_tokens=student_max_tokens,
        api_key="local" if "openai/" in student_model else os.getenv("GEMINI_API_KEY"),
        api_base=student_api_base if "openai/" in student_model else None,
        cache=False,
    )
    dspy.configure(lm=lm)

    # Reflection LM
    reflection_lm_kwargs = {
        "api_key": os.getenv("GEMINI_API_KEY") if "gemini/" in reflection_model else "local",
        "max_tokens": reflection_max_tokens,
        "cache": False,
    }

    if "gemini/" in reflection_model and use_thinking:
        reflection_lm_kwargs["thinking"] = {"type": "enabled"}

    if "openai/" in reflection_model:
        reflection_lm_kwargs["api_base"] = student_api_base
        reflection_lm_kwargs["temperature"] = 0.6

    reflection_lm = dspy.LM(reflection_model, **reflection_lm_kwargs)

    # Test connections
    console.print("🔍 Testing language model connections...")
    try:
        lm(messages=[{"role": "user", "content": "Hello"}])
        console.print(f"✅ Student LM ({student_model}) connected successfully")
    except Exception as e:
        console.print(f"❌ Student LM connection failed: {e}")
        raise

    try:
        reflection_lm(messages=[{"role": "user", "content": "What is the largest prime number below 10?"}])
        console.print(f"✅ Reflection LM ({reflection_model}) connected successfully")
    except Exception as e:
        console.print(f"❌ Reflection LM connection failed: {e}")
        raise

    return lm, reflection_lm


def prepare_datasets(
    dataset_str: str,
    test_dataset_str: str,
    train_size_ratio: float,
    max_train_examples: int,
    max_val_examples: int,
    random_seed: int,
):
    """Prepare train, validation, and test datasets."""
    try:
        from rlvr.dspy.mhqa.data import prepare_musique_dataset
    except ImportError:
        console.print("❌ Failed to import rlvr.dspy.mhqa.data")
        raise

    console.print("📚 Loading datasets...")

    # Load and split training data
    ds = prepare_musique_dataset(datasets_str=dataset_str)
    random.Random(random_seed).shuffle(ds)

    train_size = int(len(ds) * train_size_ratio)
    train_ds, val_ds = ds[:train_size], ds[train_size:]

    # Limit dataset sizes
    train_ds = train_ds[:max_train_examples]
    val_ds = val_ds[:max_val_examples]

    # Load test data
    test_ds = prepare_musique_dataset(datasets_str=test_dataset_str)

    console.print(f"📊 Dataset sizes - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    return train_ds, val_ds, test_ds


@app.command()
def evaluate(
    # Model configuration
    student_model: str = typer.Option(
        "openai/Qwen/Qwen3-8B", "--student-model", help="Student language model to evaluate"
    ),
    student_temperature: float = typer.Option(0.6, "--student-temperature", help="Temperature for student model"),
    student_max_tokens: int = typer.Option(16384, "--student-max-tokens", help="Max tokens for student model"),
    student_api_base: str = typer.Option(
        "http://0.0.0.0:8001/v1", "--student-api-base", help="API base URL for OpenAI-compatible models"
    ),
    reflection_model: str = typer.Option(
        "gemini/gemini-2.5-pro", "--reflection-model", help="Reflection language model"
    ),
    reflection_max_tokens: int = typer.Option(16384, "--reflection-max-tokens", help="Max tokens for reflection model"),
    use_thinking: bool = typer.Option(
        True, "--use-thinking/--no-thinking", help="Enable thinking mode for Gemini models"
    ),
    # Dataset configuration
    dataset_str: str = typer.Option(
        "bdsaglam/musique-mini,answerable,train", "--dataset", help="Training dataset specification"
    ),
    test_dataset_str: str = typer.Option(
        "bdsaglam/musique-mini,answerable,validation[:50]", "--test-dataset", help="Test dataset specification"
    ),
    train_size_ratio: float = typer.Option(
        0.80, "--train-ratio", help="Ratio of data to use for training vs validation"
    ),
    max_train_examples: int = typer.Option(30, "--max-train", help="Maximum number of training examples"),
    max_val_examples: int = typer.Option(30, "--max-val", help="Maximum number of validation examples"),
    random_seed: int = typer.Option(89, "--seed", help="Random seed for data shuffling"),
    # Evaluation configuration
    prompt_technique: str = typer.Option("cot", "--prompt-technique", help="Prompt technique to use (cot, etc.)"),
    num_threads: int = typer.Option(16, "--num-threads", help="Number of threads for evaluation"),
    # Output configuration
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Output directory (defaults to timestamped dir)"
    ),
    experiment_name: str = typer.Option("dspy-gepa-musique-evals", "--experiment", help="MLflow experiment name"),
):
    """Evaluate DSPy Multi-Hop QA model on MuSiQue dataset."""

    # Set up output directory
    if output_dir is None:
        exp_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"../outputs/dspy-gepa-musique-evals/{exp_id}")

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"📁 Output directory: {output_dir}")

    # Set up MLflow
    mlflow_exp = setup_mlflow(experiment_name)

    # Configure language models
    lm, reflection_lm = setup_language_models(
        student_model,
        student_temperature,
        student_max_tokens,
        student_api_base,
        reflection_model,
        reflection_max_tokens,
        use_thinking,
    )

    # Tag models in MLflow
    mlflow_exp._tags["student_lm"] = {"model": lm.model}
    mlflow_exp._tags["reflection_lm"] = {"model": reflection_lm.model}

    # Prepare datasets
    train_ds, val_ds, test_ds = prepare_datasets(
        dataset_str, test_dataset_str, train_size_ratio, max_train_examples, max_val_examples, random_seed
    )

    console.print(f"🧠 Initializing MultiHopQA with {prompt_technique} prompting...")
    program = MultiHopQA(prompt_technique=prompt_technique)

    # Evaluate on test set
    console.print("📊 Evaluating on test set...")
    evaluator = dspy.Evaluate(
        devset=test_ds,
        metric=metric,
        num_threads=num_threads,
        display_table=False,
        display_progress=True,
    )

    eval_result = evaluator(program)

    # Display results
    results_table = Table(title="Evaluation Results")
    results_table.add_column("Metric", style="cyan")
    results_table.add_column("Value", style="green")

    results_table.add_row("Overall Score", f"{eval_result:.3f}")
    results_table.add_row("Test Examples", str(len(test_ds)))
    results_table.add_row("Student Model", student_model)
    results_table.add_row("Reflection Model", reflection_model)
    results_table.add_row("Prompt Technique", prompt_technique)

    console.print(results_table)

    # Save results
    results_file = output_dir / "results.txt"
    with open(results_file, "w") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n")
        f.write(f"Overall Score: {eval_result:.3f}\n")
        f.write(f"Test Examples: {len(test_ds)}\n")
        f.write(f"Student Model: {student_model}\n")
        f.write(f"Reflection Model: {reflection_model}\n")
        f.write(f"Prompt Technique: {prompt_technique}\n")
        f.write(f"Timestamp: {datetime.datetime.now()}\n")

    console.print(f"💾 Results saved to {results_file}")

    return eval_result


if __name__ == "__main__":
    app()
