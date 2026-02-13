"""Main entry point for ARC-AGI environments.

Supports multiple environment types via the `env_type` parameter:
- "repl": REPL-based environment with `python` tool (default)
- "iterative": Iterative refinement with automatic evaluation
"""

from __future__ import annotations

from datasets import Dataset, concatenate_datasets

import verifiers as vf

from .data import prepare_dataset
from .rewards import ArcAgiRubric


def _load_dataset(dataset: str | list[str], split: str) -> Dataset:
    """Load one or more datasets and concatenate them."""
    data_folders = [dataset] if isinstance(dataset, str) else list(dataset)
    datasets = [prepare_dataset(folder, split) for folder in data_folders]
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def load_environment(
    dataset: str | list[str] = "arc-prize-2025",
    split: str = "training",
    eval_dataset: str | list[str] | None = None,
    eval_split: str = "evaluation",
    reward_mode: str = "binary",
    max_turns: int = 10,
    env_type: str = "repl",
    timeout_s: float = 5.0,
    **kwargs,
) -> vf.Environment:
    """Load an ARC-AGI environment.

    Args:
        dataset: ARC data folder name, or list of folder names to concatenate
            from environments/arc_agi/data (e.g. ["arc-prize-2024", "arc-prize-2025"]).
        split: Data split (training or evaluation).
        eval_dataset: Separate ARC data folder name(s) for evaluation (optional).
        eval_split: Evaluation data split.
        reward_mode: Reward weighting - "binary", "partial", or "combined".
        max_turns: Maximum interaction turns.
        env_type: Environment type:
            - "repl": REPL-based with `python` tool
            - "iterative": Iterative refinement with automatic evaluation
        timeout_s: Timeout for code execution (iterative env only).
        **kwargs: Additional arguments passed to the environment.

    Returns:
        Configured environment instance.
    """
    train_ds = _load_dataset(dataset, split)

    eval_ds = None
    if eval_dataset is not None:
        eval_ds = _load_dataset(eval_dataset, eval_split)

    parser = vf.Parser()
    rubric = ArcAgiRubric(parser=parser, reward_mode=reward_mode)

    if env_type == "repl":
        from .envs.repl import ArcAgiREPLEnv

        env = ArcAgiREPLEnv(
            dataset=train_ds,
            eval_dataset=eval_ds,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )
    elif env_type == "iterative":
        from .envs.iterative import ArcAgiIterativeEnv

        env = ArcAgiIterativeEnv(
            dataset=train_ds,
            eval_dataset=eval_ds,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            timeout_s=timeout_s,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Must be 'repl' or 'iterative'.")

    return env
