import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import typer
import wandb
from accelerate import Accelerator
from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from tqdm import tqdm
from trl import GRPOConfig

from verifiers.envs.tool_env import ToolEnv
from verifiers.imports import LLM, SamplingParams
from verifiers.prompts import QA_TOOL_PROMPT_TEMPLATE, RETRIEVE_FEW_SHOT
from verifiers.rubrics.citation import make_citation_reward_func
from verifiers.rubrics.language import natural_language_reward_func
from verifiers.rubrics.musique import (
    musique_em_reward_func,
    musique_f1_reward_func,
    musique_supporting_recall_reward_func,
)
from verifiers.tools import make_retrieve_tool
from verifiers.tools.retrieve import make_get_tool
from verifiers.trainers.grpo_env_trainer import GRPOEnvTrainer
from verifiers.utils.cuda import get_half_precision_dtype
from verifiers.utils.logging_utils import setup_logging
from verifiers.utils.model_utils import get_model_and_tokenizer, get_tokenizer

load_dotenv()

setup_logging()

log = logging.getLogger(__name__)

app = typer.Typer()

accelerator = Accelerator()


def prepare_dataset(path: str, name: str, split: str) -> Dataset:
    ds = load_dataset(path, name, split=split)
    if "trivia_qa" in path:
        from verifiers.datasets.triviaqa import preprocess_dataset

        ds = preprocess_dataset(ds)
    elif "musique" in path:
        from verifiers.datasets.musique import preprocess_dataset

        ds = preprocess_dataset(ds)
    return ds


def prepare_datasets(dataset_str: str) -> Dataset:
    """
    Prepare a dataset from a string of the form "path,name,split".
    """
    ds_list = []
    for s in dataset_str.split(";"):
        path, name, split = s.split(",")
        ds = prepare_dataset(path, name, split)
        ds_list.append(ds)

    return concatenate_datasets(ds_list).shuffle(seed=89)


def create_environment(
    tokenizer: Any,
    retriever: str,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    n_jobs: int = 1,
    top_k: int = 1,
    few_shot_prob: float = 1.0,
):
    """
    Create and initialize the appropriate environment based on the specified type.

    Args:
        train_dataset: Dataset for training
        eval_dataset: Dataset for evaluation
        tokenizer: Tokenizer instance
        retriever: Retriever to use

    Returns:
        A tuple containing the initialized environment and a default suffix for run naming
    """
    log.info("Initializing the environment")

    vf_env = ToolEnv(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        tools=[make_retrieve_tool(name=retriever, top_k=top_k), make_get_tool()],
        system_prompt=QA_TOOL_PROMPT_TEMPLATE,
        few_shot=RETRIEVE_FEW_SHOT[0],
        few_shot_prob=few_shot_prob,
        max_steps=50,
        n_jobs=n_jobs,
    )

    return vf_env


def get_model_name(model_path: str) -> str:
    if Path(model_path).exists():
        with open(Path(model_path) / "config.json", "r") as f:
            config = json.load(f)
        return config["_name_or_path"].split("/")[-1]
    else:
        return model_path.split("/")[-1]


@app.command("train")
def train(
    model_path: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct", "--model"),
    datasets_str: str = typer.Option("bdsaglam/musique,answerable,train", "--datasets"),
    noise_rate: float = typer.Option(1.0, help="Noise rate to use"),
    retriever: str = typer.Option("hybrid", help="Retriever to use"),
    retriever_top_k: int = typer.Option(1, help="Number of retriever results to use"),
    few_shot_prob: float = typer.Option(0.0, help="Probability of using few-shot examples"),
    n_env_jobs: int = typer.Option(1, help="Number of environments to run in parallel"),
    max_prompt_length: int = typer.Option(4096),
    max_completion_length: int = typer.Option(1024),
    temperature: float = typer.Option(0.5),
    num_generations: int = typer.Option(8),
    scale_rewards: bool = typer.Option(False, help="Scale rewards"),
    kl_beta: float = typer.Option(0.04, help="KL beta"),
    batch_size: int = typer.Option(32),
    gradient_accumulation_steps: int = typer.Option(2),
    learning_rate: float = typer.Option(1e-6),
    peft: bool = typer.Option(True),
    lora_r: int = typer.Option(32, help="LORA rank"),
    lora_alpha: int = typer.Option(64, help="LORA alpha"),
    n_epochs: int = typer.Option(2),
    report_to: str = typer.Option("wandb", help="Report to wandb"),
    push_to_hub: bool = typer.Option(True, help="Push to hub"),
    out: Path = typer.Option("./outputs/"),
    run_name: str | None = None,
    resume_from_checkpoint: str | None = typer.Option(None, help="Resume training from a checkpoint"),
):
    """Train a model using GRPO for code generation or tool use."""

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # Load dataset
    train_dataset = prepare_datasets(datasets_str)
    train_dataset = train_dataset.map(
        lambda x: {"docs": [doc for doc in x["docs"] if doc["is_supporting"] or random.random() < noise_rate]}
    )
    log.info(f"Train dataset: {len(train_dataset)}")

    # Load model and tokenizer
    model, tokenizer = get_model_and_tokenizer(model_path)

    # Initialize environment based on env_type
    vf_env = create_environment(
        tokenizer=tokenizer,
        retriever=retriever,
        train_dataset=train_dataset,
        n_jobs=n_env_jobs,
        top_k=retriever_top_k,
        few_shot_prob=few_shot_prob,
    )

    # Use provided suffix or default based on env_type
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{get_model_name(model_path)}-ragent-grpo-{timestamp}"

    training_args = GRPOConfig(
        output_dir=out / run_name,
        push_to_hub=push_to_hub,
        hub_model_id=run_name,
        bf16=True,
        num_train_epochs=n_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.1,
        num_iterations=2,  # steps per global batch (1 on-policy, 1 off-policy)
        num_generations=num_generations,
        temperature=temperature,
        beta=kl_beta,
        reward_weights=None,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_gpu_memory_utilization=0.5,
        save_strategy="steps",
        save_steps=100,
        save_only_model=False,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to=report_to,
        run_name=run_name,
        eval_strategy="no",
        eval_on_start=False,
        per_device_eval_batch_size=batch_size,
        eval_accumulation_steps=1,
    )
    # Configure LoRA
    if peft:
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "up_proj",
                "down_proj",
                "gate_proj",
            ],
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Initialize trainer
    reward_funcs = [
        musique_em_reward_func,
        musique_f1_reward_func,
        musique_supporting_recall_reward_func,
        make_citation_reward_func(cite_tag="cite"),
        natural_language_reward_func,
        *vf_env.get_reward_funcs(),
    ]
    trainer = GRPOEnvTrainer(
        model=model,
        processing_class=tokenizer,
        peft_config=peft_config,
        env=vf_env,
        reward_funcs=reward_funcs,
        scale_rewards=scale_rewards,
        args=training_args,
        train_dataset=vf_env.get_dataset(),
        eval_dataset=vf_env.get_eval_dataset(),
    )

    # Start training
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Update experiment configs
    if wandb.run is not None:
        wandb.run.config.update(
            {
                "datasets": datasets_str,
                "noise_rate": noise_rate,
                "retriever": retriever,
                "retriever_top_k": retriever_top_k,
                "few_shot_prob": few_shot_prob,
                "n_env_jobs": n_env_jobs,
                "max_prompt_length": max_prompt_length,
                "max_completion_length": max_completion_length,
                "num_generations": num_generations,
                "scale_rewards": scale_rewards,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "learning_rate": learning_rate,
                "peft": peft,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "n_epochs": n_epochs,
            }
        )

    # Cleanup
    wandb.finish()
    del model
    del trainer
    torch.cuda.empty_cache()


@app.command("predict")
def predict(
    model_path: str = typer.Option("meta-llama/Llama-3.1-8B-Instruct", "--model"),
    dataset_path: str = typer.Option("bdsaglam/musique-mini"),
    dataset_name: str = typer.Option("answerable"),
    dataset_split: str = typer.Option("validation"),
    retriever: str = typer.Option("bm25", help="Retriever to use"),
    retriever_top_k: int = typer.Option(1, help="Number of retriever results to use"),
    few_shot_prob: float = typer.Option(0.0, help="Probability of using few-shot examples"),
    n_env_jobs: int = typer.Option(32, help="Number of environments to run in parallel"),
    batch_size: int = typer.Option(32, "--batch-size", "-bs"),
    max_completion_length: int = typer.Option(1024, "-cl", "--max-completion-length"),
    temperature: float = typer.Option(0.5),
    top_p: float = typer.Option(0.95),
    repeat: int = typer.Option(1, help="Number of times to repeat the trajectory"),
    out: Path = typer.Option("./outputs/predictions.jsonl"),
    seed: int = 89,
):
    """Predict with a model on a dataset using RAG-based verification."""
    if n_env_jobs > 1 and retriever == "bm25":
        raise ValueError("BM25 does not support parallel environments. Run rerank service and use 'lexical', instead.")

    # Load dataset
    dataset = prepare_datasets(f"{dataset_path},{dataset_name},{dataset_split}")
    log.info(f"Dataset: {len(dataset)}")

    # Load model and tokenizer
    tokenizer_path = "meta-llama/Llama-3.1-8B-Instruct" if "openai" in model_path else model_path
    tokenizer = get_tokenizer(tokenizer_path)

    # Initialize environment
    vf_env = create_environment(
        tokenizer=tokenizer,
        train_dataset=dataset,
        n_jobs=n_env_jobs,
        retriever=retriever,
        top_k=retriever_top_k,
        few_shot_prob=few_shot_prob,
    )

    # Initialize vLLM for serving
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=get_half_precision_dtype(),
        gpu_memory_utilization=0.60,
        tensor_parallel_size=os.getenv("CUDA_VISIBLE_DEVICES", "0").count(",") + 1,
        seed=seed,
        max_model_len=8192 * 2,
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        max_tokens=max_completion_length,
        stop=[
            *vf_env.special_stop_tokens,
            tokenizer.eos_token,
            tokenizer.pad_token,
        ],
        temperature=temperature,
        top_p=top_p,
    )

    # Process dataset in batches with progress bar
    ds = vf_env.get_dataset().repeat(repeat)
    records = []
    for i in tqdm(range(0, len(ds), batch_size), desc="Processing batches", total=len(ds) // batch_size):
        inputs = ds.select(range(i, min(i + batch_size, len(ds))))

        # Generate completions and interact with environment
        result = vf_env.generate(
            inputs=inputs,
            llm=llm,
            sampling_params=sampling_params,
        )

        # Store trajectories
        for input_data, messages in zip(inputs, result["messages"]):
            record = {
                **input_data,
                "trajectory": input_data["prompt"] + messages,
            }
            records.append(record)

    # Save trajectories to jsonl file
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    log.info(f"Saved predictions to {out}")


if __name__ == "__main__":
    app()
