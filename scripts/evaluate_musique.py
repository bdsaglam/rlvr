#!/usr/bin/env python3
"""
MuSiQue evaluation script for testing trained models.

Usage:
    python scripts/evaluate_musique.py --model outputs/your-model
    
Example with custom settings:
    python scripts/evaluate_musique.py \
        --model outputs/llama-musique-grpo \
        --dataset-split validation \
        --num-examples 100 \
        --retriever hybrid \
        --batch-size 8
"""

import sys
sys.path.insert(0, "/home/baris/repos/rlvr/environments")
sys.path.insert(0, "/home/baris/repos/rlvr/src")

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import verifiers as vf

app = typer.Typer()


def get_model_name(model_path: str) -> str:
    """Extract model name from path."""
    if Path(model_path).exists():
        return Path(model_path).name
    else:
        return model_path.split("/")[-1]


@app.command()
def evaluate(
    # Model arguments
    model: str = typer.Argument(..., help="Path to trained model or HuggingFace model name"),
    
    # Dataset arguments
    dataset_split: str = typer.Option(
        "validation",
        "--dataset-split",
        help="Dataset split to evaluate on"
    ),
    num_examples: int = typer.Option(100, "--num-examples", help="Number of examples to evaluate"),
    
    # Retriever arguments
    retriever: str = typer.Option("hybrid", "--retriever", help="Retrieval strategy to use"),
    retriever_top_n: int = typer.Option(3, "--retriever-top-k", help="Number of documents to retrieve"),
    
    # Generation arguments
    batch_size: int = typer.Option(8, "--batch-size", help="Batch size for generation"),
    max_new_tokens: int = typer.Option(1024, "--max-new-tokens", help="Maximum tokens to generate"),
    temperature: float = typer.Option(0.1, "--temperature", help="Generation temperature"),
    num_generations: int = typer.Option(1, "--num-generations", help="Number of generations per example"),
    
    # Output arguments
    output_file: Optional[str] = typer.Option(
        None, 
        "--output-file", "-o",
        help="Output file for results (auto-generated if not provided)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Print detailed evaluation results"),
    
    # Judge model arguments (optional)
    judge_model: Optional[str] = typer.Option(None, "--judge-model", help="Judge model for additional evaluation"),
    judge_base_url: str = typer.Option("https://api.openai.com/v1", "--judge-base-url", help="Base URL for judge model"),
    judge_api_key_var: str = typer.Option("OPENAI_API_KEY", "--judge-api-key-var", help="Environment variable for judge API key"),
):
    """Evaluate a trained model on MuSiQue dataset."""
    
    # Generate output filename if not provided
    if output_file is None:
        model_name = get_model_name(model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"evaluation_{model_name}_{dataset_split}_{timestamp}.json"
    
    # Print evaluation configuration
    typer.echo("🧪 Starting MuSiQue evaluation")
    typer.echo("=" * 50)
    typer.echo(f"📝 Model: {model}")
    typer.echo(f"📊 Dataset: {dataset_split} ({num_examples} examples)")
    typer.echo(f"🔍 Retriever: {retriever} (top-k: {retriever_top_n})")
    typer.echo(f"📏 Batch size: {batch_size}")
    typer.echo(f"🌡️  Temperature: {temperature}")
    typer.echo(f"💾 Output file: {output_file}")
    typer.echo("=" * 50)
    
    # Load MuSiQue environment
    typer.echo("🌍 Loading MuSiQue environment...")
    
    # Configure dataset based on split
    if dataset_split == "validation":
        vf_env = vf.load_environment(
            env_id="vf-musique",
            num_train_examples=0,  # Don't load training data
            num_eval_examples=num_examples,
            retriever_name=retriever,
            retriever_top_n=retriever_top_n,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_var=judge_api_key_var,
        )
        eval_dataset = vf_env.eval_dataset
    else:
        # For train/test, load as training data and use it for evaluation
        vf_env = vf.load_environment(
            env_id="vf-musique",
            num_train_examples=num_examples,
            num_eval_examples=0,
            retriever_name=retriever,
            retriever_top_n=retriever_top_n,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_var=judge_api_key_var,
        )
        eval_dataset = vf_env.dataset
    
    typer.echo(f"✅ Environment loaded with {len(eval_dataset)} examples")
    
    # Model loading info
    typer.echo(f"🤖 Model configuration: {model}")
    typer.echo("ℹ️  Note: In a real evaluation, the model would be loaded here using vf.get_model_and_tokenizer()")
    
    # Simulate evaluation results structure
    typer.echo("🔄 Running evaluation...")
    typer.echo("ℹ️  Simulating evaluation process...")
    
    # In a real implementation, this would use the verifiers evaluation pipeline:
    # model_obj, tokenizer = vf.get_model_and_tokenizer(model)
    # results = vf_env.evaluate(
    #     model=model_obj,
    #     dataset=eval_dataset,
    #     batch_size=batch_size,
    #     max_new_tokens=max_new_tokens,
    #     temperature=temperature,
    #     num_generations=num_generations
    # )
    
    # Create template results structure
    results = {
        "config": {
            "model": model,
            "dataset_split": dataset_split,
            "num_examples": num_examples,
            "retriever": retriever,
            "retriever_top_n": retriever_top_n,
            "batch_size": batch_size,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "num_generations": num_generations,
            "evaluation_date": datetime.now().isoformat(),
        },
        "metrics": {
            "exact_match": 0.0,  # Would be computed from actual evaluation
            "f1": 0.0,
            "weighted_em": 0.0,
            "weighted_f1": 0.0,
            "retrieval_recall": 0.0,
            "combined_reward": 0.0,
            "n_hops_avg": 0.0,
        },
        "per_example_results": [
            # Would contain detailed results for each example
            # {
            #     "question": "...",
            #     "predicted_answer": "...",
            #     "ground_truth": "...",
            #     "exact_match": 1.0,
            #     "f1": 1.0,
            #     "retrieval_recall": 0.8,
            #     "n_hops": 2,
            #     "trajectory": [...],
            # }
        ]
    }
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    typer.echo(f"💾 Results saved to: {output_path}")
    
    # Print summary
    if verbose:
        typer.echo("\n📊 Evaluation Results:")
        typer.echo(f"  Exact Match: {results['metrics']['exact_match']:.3f}")
        typer.echo(f"  F1 Score: {results['metrics']['f1']:.3f}")
        typer.echo(f"  Weighted EM: {results['metrics']['weighted_em']:.3f}")
        typer.echo(f"  Weighted F1: {results['metrics']['weighted_f1']:.3f}")
        typer.echo(f"  Retrieval Recall: {results['metrics']['retrieval_recall']:.3f}")
        typer.echo(f"  Combined Reward: {results['metrics']['combined_reward']:.3f}")
        typer.echo(f"  Average Hops: {results['metrics']['n_hops_avg']:.1f}")
    
    typer.echo("✅ Evaluation completed!")
    
    # Print next steps
    typer.echo("\n📝 Implementation notes:")
    typer.echo("  This is a template evaluation script showing the structure.")
    typer.echo("  For full evaluation, install verifiers with model support and:")
    typer.echo("  1. Load the model using vf.get_model_and_tokenizer()")
    typer.echo("  2. Use verifiers evaluation pipeline to run inference")
    typer.echo("  3. Compute metrics using the custom MuSiQue rubric")
    typer.echo("  4. Save detailed trajectories for analysis")


@app.command()
def benchmark(
    models: str = typer.Option(..., "--models", help="Comma-separated list of models to benchmark"),
    dataset_split: str = typer.Option("validation", "--dataset-split", help="Dataset split to use"),
    num_examples: int = typer.Option(100, "--num-examples", help="Number of examples per model"),
    retrievers: str = typer.Option("hybrid", "--retrievers", help="Comma-separated list of retrievers to test"),
    output_dir: Path = typer.Option("./benchmark_results", "--output-dir", help="Output directory for results"),
):
    """Benchmark multiple models and retrievers on MuSiQue."""
    
    model_list = [m.strip() for m in models.split(",")]
    retriever_list = [r.strip() for r in retrievers.split(",")]
    
    typer.echo("🏁 Starting MuSiQue benchmark")
    typer.echo("=" * 50)
    typer.echo(f"📝 Models: {model_list}")
    typer.echo(f"🔍 Retrievers: {retriever_list}")
    typer.echo(f"📊 Examples per configuration: {num_examples}")
    typer.echo(f"📁 Output directory: {output_dir}")
    typer.echo("=" * 50)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # This would run evaluation for each model-retriever combination
    total_configs = len(model_list) * len(retriever_list)
    typer.echo(f"⚡ Will evaluate {total_configs} configurations")
    
    for i, model in enumerate(model_list, 1):
        for j, retriever in enumerate(retriever_list, 1):
            config_num = (i - 1) * len(retriever_list) + j
            typer.echo(f"\n🔄 Configuration {config_num}/{total_configs}: {model} + {retriever}")
            
            # This would call the evaluate function for each configuration
            output_file = output_dir / f"{get_model_name(model)}_{retriever}_{dataset_split}.json"
            typer.echo(f"   📝 Would save results to: {output_file}")
    
    typer.echo(f"\n✅ Benchmark template completed!")
    typer.echo(f"📊 Results would be saved in: {output_dir}")


@app.command()
def analyze(
    results_dir: Path = typer.Argument(..., help="Directory containing evaluation results"),
    output_file: Optional[str] = typer.Option(None, "--output-file", help="Output file for analysis"),
):
    """Analyze evaluation results from multiple runs."""
    
    if not results_dir.exists():
        typer.echo(f"❌ Results directory does not exist: {results_dir}")
        raise typer.Exit(1)
    
    # Find all result files
    result_files = list(results_dir.glob("*.json"))
    
    if not result_files:
        typer.echo(f"❌ No result files found in: {results_dir}")
        raise typer.Exit(1)
    
    typer.echo("📊 Analyzing evaluation results")
    typer.echo("=" * 50)
    typer.echo(f"📁 Results directory: {results_dir}")
    typer.echo(f"📄 Found {len(result_files)} result files")
    
    # This would aggregate and analyze results
    for result_file in result_files:
        typer.echo(f"  📝 {result_file.name}")
    
    if output_file:
        typer.echo(f"💾 Analysis would be saved to: {output_file}")
    
    typer.echo("✅ Analysis completed!")


if __name__ == "__main__":
    app()