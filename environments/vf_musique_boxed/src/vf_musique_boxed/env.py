from textwrap import dedent

import verifiers as vf
from agents import RunContextWrapper
from verifiers.envs.stateful_tool_env import StatefulToolEnv
from verifiers.types import Messages, State
from verifiers.utils.data_utils import extract_boxed_answer

from .data import prepare_dataset
from .rewards import (
    combined_reward,
    exact_match_reward,
    f1_reward,
    format_reward,
    retrieval_precision_reward,
    retrieval_recall_reward,
)
from .tools import make_get_tool, make_retrieve_tool


class MuSiQueEnv(StatefulToolEnv):
    """Custom ToolEnv for MuSiQue with document injection via tool_args."""

    def update_tool_args(self, tool_args: dict, messages: Messages, state: State, **kwargs) -> dict:
        """Update tool_args with the current state."""
        tool_args["wrapper"] = RunContextWrapper(context={"info": state["info"]})
        return tool_args


# Custom rubric for MuSiQue evaluation
def MuSiQueRubric(parser, **kwargs):
    """Create MuSiQue rubric with all reward functions."""
    reward_funcs = [
        exact_match_reward,
        f1_reward,
        retrieval_recall_reward,
        retrieval_precision_reward,
        format_reward,
        combined_reward,
    ]

    # Combined reward gets weight 1, others are for metrics only
    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    return vf.Rubric(funcs=reward_funcs, weights=weights, parser=parser, **kwargs)


def load_environment(
    datasets_str: str = "bdsaglam/musique,answerable,train",
    eval_datasets_str: str | None = None,
    noise_rate: float = 1.0,
    retriever: str = "hybrid",
    **kwargs,
) -> vf.Environment:
    """Load MuSiQue environment for multi-hop question answering."""

    # Load dataset from datasets_str
    dataset = prepare_dataset(datasets_str, noise_rate=noise_rate)

    # Load evaluation dataset if specified
    eval_dataset = None
    if eval_datasets_str:
        eval_dataset = prepare_dataset(eval_datasets_str, noise_rate=noise_rate)

    # Create tools
    tools = [
        make_retrieve_tool(name=retriever),
    ]

    # System prompt for MuSiQue
    system_prompt = dedent("""
    Answer the question based on the information provided by tools.

    For each step:
    1. Think about the question and the information provided by the tools. Plan next action.
    2. Use `retrieve_documents` tool to retrieve documents
    3. Continue until you find the answer through multi-hop reasoning. The question is answerable from the docs. 
    4. In the **last** step:
        - Reflect on your previous steps 
        - Give your final answer inside `\\boxed{...}`
    An example for your final message:
    ```
    [your thinking and explanation here]
    Answer: \\boxed{...}
    ```

    - Do not make up tools or arguments that aren't listed.
    - Make one tool call per step.
    - Questions require multi-hop reasoning across multiple documents.
    - Continue searching until you find all necessary information to answer the question.
    """).strip()

    # Create parser
    parser = vf.Parser(
        extract_fn=extract_boxed_answer,
    )

    # Create environment
    env = MuSiQueEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        tools=tools,
        parser=parser,
        rubric=MuSiQueRubric(parser=parser),
        max_turns=10,
        **kwargs,
    )

    return env
