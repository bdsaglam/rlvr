"""Main entry point for ARC-AGI environments.

Supports multiple environment types via the `env_type` parameter:
- "repl": REPL-based environment with `python` tool (default)
- "iterative": Iterative refinement with automatic evaluation
"""

from __future__ import annotations

import verifiers as vf

from .data import prepare_dataset
from .rewards import ArcAgiRubric


def load_environment(
    data_dir: str = "data/arc-prize-2025",
    split: str = "training",
    eval_data_dir: str | None = None,
    eval_split: str = "evaluation",
    reward_mode: str = "binary",
    max_turns: int = 10,
    env_type: str = "iterative",
    timeout_s: float = 5.0,
    **kwargs,
) -> vf.Environment:
    """Load an ARC-AGI environment.

    Args:
        data_dir: Path to ARC data directory.
        split: Data split (training or evaluation).
        eval_data_dir: Separate data dir for evaluation (optional).
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
    dataset = prepare_dataset(data_dir, split)

    eval_dataset = None
    if eval_data_dir is not None:
        eval_dataset = prepare_dataset(eval_data_dir, eval_split)

    parser = vf.Parser()
    rubric = ArcAgiRubric(parser=parser, reward_mode=reward_mode)

    if env_type == "repl":
        from .envs.repl import ArcAgiREPLEnv

        env = ArcAgiREPLEnv(
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )
    elif env_type == "iterative":
        from .envs.iterative import ArcAgiIterativeEnv

        env = ArcAgiIterativeEnv(
            dataset=dataset,
            eval_dataset=eval_dataset,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            timeout_s=timeout_s,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown env_type: {env_type}. Must be 'repl' or 'iterative'.")

    return env
