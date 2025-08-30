#!/usr/bin/env python
"""Train a model on MuSiQue using VERL library with GRPO."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import typer
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

import verl
from verl.algorithms import GRPOConfig
from verl.trainer import Trainer
from verl.utils.reward_score import RewardScore

assert load_dotenv(), "Failed to load .env file"

app = typer.Typer()


def get_model_name(model_path: str) -> str:
    """Extract model name from path."""
    if Path(model_path).exists():
        return Path(model_path).name
    else:
        return model_path.split("/")[-1]


class MuSiQueRewardScore(RewardScore):
    """Custom reward score for MuSiQue multi-hop QA."""
    
    def __init__(self, noise_rate: float = 1.0):
        super().__init__()
        self.noise_rate = noise_rate
    
    def compute_score(
        self,
        prompts: List[str],
        responses: List[str],
        ground_truths: List[Dict[str, Any]],
    ) -> List[float]:
        """Compute rewards based on answer correctness and retrieval quality."""
        rewards = []
        
        for response, gt in zip(responses, ground_truths):
            # Extract answer from response
            answer = self._extract_answer(response)
            
            # Compute F1 score for answer
            f1_score = self._compute_f1(answer, gt.get("answer", ""))
            
            # For now, just use F1 score as reward
            # In full implementation, would also consider retrieval quality
            rewards.append(f1_score)
        
        return rewards
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from model response."""
        # Simple extraction - look for "Answer:" pattern
        if "Answer:" in response:
            answer = response.split("Answer:")[-1].strip()
            # Clean up any trailing punctuation or whitespace
            answer = answer.split("\n")[0].strip()
            return answer
        return response.strip()
    
    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute F1 score between prediction and ground truth."""
        if not prediction or not ground_truth:
            return 0.0
        
        pred_tokens = prediction.lower().split()
        gt_tokens = ground_truth.lower().split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common = set(pred_tokens) & set(gt_tokens)
        
        if len(common) == 0:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1


def prepare_dataset(
    datasets_str: str,
    max_examples: Optional[int] = None,
) -> Any:
    """Prepare MuSiQue dataset for training."""
    # Parse dataset string
    parts = datasets_str.split(",")
    if len(parts) == 3:
        dataset_name, config_name, split = parts
    elif len(parts) == 2:
        dataset_name, split = parts
        config_name = None
    else:
        dataset_name = parts[0]
        config_name = None
        split = "train"
    
    # Load dataset
    if config_name:
        dataset = load_dataset(dataset_name, config_name, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    # Limit examples if specified
    if max_examples:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
    
    return dataset


def format_prompt(example: Dict[str, Any]) -> str:
    """Format MuSiQue example into prompt."""
    question = example.get("question", "")
    
    # Build context from paragraphs
    context_parts = []
    if "paragraphs" in example:
        for para in example["paragraphs"]:
            title = para.get("title", "")
            text = para.get("paragraph_text", "")
            if title and text:
                context_parts.append(f"{title}: {text}")
    
    context = "\n\n".join(context_parts) if context_parts else ""
    
    # Format prompt
    if context:
        prompt = f"""Context:
{context}

Question: {question}

Please answer the question based on the given context. Think step by step.

Answer:"""
    else:
        prompt = f"""Question: {question}

Please answer the question. Think step by step.

Answer:"""
    
    return prompt


@app.command()
def train(
    # Model arguments
    model: str = typer.Option(
        "Qwen/Qwen2.5-7B-Instruct",
        "--model", "-m",
        help="Model path or HuggingFace model name"
    ),
    # Dataset arguments
    datasets_str: str = typer.Option(
        "bdsaglam/musique,answerable,train",
        "--datasets",
        help="Datasets string in format 'path,name,split'"
    ),
    max_train_examples: Optional[int] = typer.Option(
        None,
        help="Maximum number of training examples"
    ),
    noise_rate: float = typer.Option(
        1.0,
        help="Noise rate for filtering non-supporting documents"
    ),
    # Training arguments
    num_epochs: int = typer.Option(1, help="Number of training epochs"),
    batch_size: int = typer.Option(8, help="Global batch size"),
    micro_batch_size: int = typer.Option(1, help="Micro batch size per device"),
    num_generations: int = typer.Option(8, help="Number of generations per prompt"),
    learning_rate: float = typer.Option(1e-6, help="Learning rate"),
    # GRPO arguments
    kl_beta: float = typer.Option(0.04, help="KL divergence coefficient"),
    # Generation arguments
    max_prompt_length: int = typer.Option(4096, help="Maximum prompt length"),
    max_new_tokens: int = typer.Option(1024, help="Maximum new tokens"),
    temperature: float = typer.Option(0.5, help="Generation temperature"),
    # Output arguments
    output_dir: Path = typer.Option("./outputs", help="Output directory"),
    run_name: Optional[str] = typer.Option(None, help="Run name"),
    # Hardware arguments
    num_gpus: int = typer.Option(1, help="Number of GPUs to use"),
):
    """Train a model on MuSiQue using VERL with GRPO."""
    
    # Generate run name if not provided
    if run_name is None:
        model_name = get_model_name(model)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_name}-musique-verl-grpo-{timestamp}"
    
    # Print configuration
    typer.echo("🚀 Starting MuSiQue training with VERL")
    typer.echo("=" * 50)
    typer.echo(f"📝 Model: {model}")
    typer.echo(f"🏷️  Run name: {run_name}")
    typer.echo(f"📊 Dataset: {datasets_str}")
    typer.echo(f"🎲 Noise rate: {noise_rate}")
    typer.echo(f"📏 Batch size: {batch_size}")
    typer.echo(f"🔢 Generations: {num_generations}")
    typer.echo(f"📈 Learning rate: {learning_rate}")
    typer.echo(f"🔄 Epochs: {num_epochs}")
    typer.echo(f"🖥️  GPUs: {num_gpus}")
    typer.echo("=" * 50)
    
    # Load dataset
    typer.echo("📚 Loading dataset...")
    dataset = prepare_dataset(datasets_str, max_train_examples)
    typer.echo(f"✅ Loaded {len(dataset)} examples")
    
    # Format prompts
    typer.echo("📝 Formatting prompts...")
    prompts = [format_prompt(ex) for ex in dataset]
    ground_truths = [
        {"answer": ex.get("answer", ""), "supporting_facts": ex.get("supporting_facts", [])}
        for ex in dataset
    ]
    
    # Load model and tokenizer
    typer.echo(f"🤖 Loading model: {model}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    typer.echo("✅ Model and tokenizer loaded")
    
    # Create reward scorer
    typer.echo("🎯 Setting up reward scorer...")
    reward_scorer = MuSiQueRewardScore(noise_rate=noise_rate)
    
    # Create GRPO configuration
    typer.echo("⚙️  Configuring GRPO...")
    grpo_config = GRPOConfig(
        # Training parameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=batch_size // (micro_batch_size * num_gpus),
        learning_rate=learning_rate,
        warmup_steps=10,
        # GRPO parameters
        num_generations=num_generations,
        kl_coef=kl_beta,
        # Generation parameters
        max_prompt_length=max_prompt_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=0.95,
        # Output
        output_dir=str(output_dir / run_name),
        logging_steps=1,
        save_steps=100,
        save_strategy="steps",
        report_to="console",  # Can be changed to "wandb"
        run_name=run_name,
    )
    
    # Create trainer
    typer.echo("🏃 Creating GRPO trainer...")
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        config=grpo_config,
        reward_scorer=reward_scorer,
        train_prompts=prompts,
        train_ground_truths=ground_truths,
    )
    
    # Start training
    typer.echo("\n🔥 Starting training!")
    typer.echo("=" * 50)
    
    try:
        trainer.train()
        typer.echo("\n✅ Training completed successfully!")
        
        # Save final model
        final_model_path = output_dir / run_name / "final"
        trainer.save_model(final_model_path)
        typer.echo(f"💾 Model saved to: {final_model_path}")
        
        # Print next steps
        typer.echo("\n📝 Next steps:")
        typer.echo(f"   1. Find your model in: {output_dir / run_name}")
        typer.echo(f"   2. Evaluate with: python scripts/evaluate_musique.py --model {final_model_path}")
        
    except KeyboardInterrupt:
        typer.echo("\n⚠️  Training interrupted by user")
        typer.echo(f"💾 Checkpoints saved in: {output_dir / run_name}")
        
    except Exception as e:
        typer.echo(f"\n❌ Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        del model
        del trainer
        torch.cuda.empty_cache()


@app.command()
def test(
    # Test parameters
    model: str = typer.Option(
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--model",
        help="Small model for testing"
    ),
    num_examples: int = typer.Option(10, help="Number of examples to test"),
    num_generations: int = typer.Option(2, help="Generations per prompt"),
):
    """Quick test of the training setup with minimal resources."""
    
    typer.echo("🧪 Running quick test...")
    
    # Use minimal parameters for testing
    train(
        model=model,
        datasets_str="bdsaglam/musique,answerable,train",
        max_train_examples=num_examples,
        num_epochs=1,
        batch_size=2,
        micro_batch_size=1,
        num_generations=num_generations,
        learning_rate=1e-5,
        output_dir=Path("./outputs/test"),
        run_name="musique-verl-test",
        num_gpus=1,
    )


if __name__ == "__main__":
    app()