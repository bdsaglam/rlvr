from textwrap import dedent
from typing import Any

import verifiers as vf
from agents import RunContextWrapper
from verifiers.envs.stateful_tool_env import StatefulToolEnv
from verifiers.types import Messages, State

from .data import prepare_dataset
from .rewards import (
    citation_reward,
    combined_reward,
    exact_match_reward,
    f1_reward,
    format_reward,
)
from .sub_agent import make_sub_agent_tool
from .tools import complete


class MuSiQueEnv(StatefulToolEnv):
    """Custom ToolEnv for MuSiQue with document injection via tool_args."""

    async def is_completed(self, messages: Messages, state: State, **kwargs: Any) -> bool:
        completed = await super().is_completed(messages, state, **kwargs)
        if completed:
            return True
        if messages:
            for tool_call in messages[-1].get("tool_calls", []):
                for tool_call in messages[-1]["tool_calls"]:
                    tool_name: str = tool_call.function.name
                    if tool_name == "complete":
                        return True
        return False

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
        citation_reward,
        format_reward,
        combined_reward,
    ]

    # Combined reward gets weight 1, others are for metrics only
    weights = [0.0, 0.0, 0.0, 0.0, 1.0]

    return vf.Rubric(funcs=reward_funcs, weights=weights, parser=parser, **kwargs)


def load_environment(
    datasets_str: str = "bdsaglam/musique,answerable,train",
    eval_datasets_str: str | None = None,
    noise_rate: float = 1.0,
    retriever: str = "hybrid",
    sub_agent_model: str | None = None,
    **kwargs,
) -> vf.Environment:
    """Load MuSiQue environment for multi-hop question answering."""

    # Load dataset from datasets_str
    dataset = prepare_dataset(datasets_str, noise_rate=noise_rate)

    # Load evaluation dataset if specified
    eval_dataset = None
    if eval_datasets_str:
        eval_dataset = prepare_dataset(eval_datasets_str, noise_rate=noise_rate)

    # Create tools - using sub-agent architecture
    answer_subquestion = make_sub_agent_tool(retriever=retriever, model=sub_agent_model)
    tools = [
        answer_subquestion,
        complete,
    ]

    # System prompt for MuSiQue - Orchestrator Agent
    system_prompt = dedent("""
    You are an orchestrator agent for multi-hop question answering. Your role is to plan and coordinate the reasoning process, delegating specific sub-questions to a specialized sub-agent.

    **Your Process:**
    1. **Plan**: Think about the multi-hop reasoning strategy
    2. **Delegate**: Break down the main question into focused sub-questions
    3. **Sub-questions**: Use the `answer_subquestion` tool for each sub-question that requires document retrieval
    4. **Synthesize**: Combine the sub-answers to form your final reasoning
    5. **Complete**: Call `complete` tool with your final answer

    **Important Guidelines:**
    - You do NOT directly access documents or retrieval tools
    - The `answer_subquestion` tool handles all document retrieval and citation
    - Focus on high-level reasoning and coordination
    - Each sub-question should be specific and focused
    - Questions require multi-hop reasoning across multiple documents
    - Make one tool call per step

    **Final Step Format:**
    When you have gathered enough information, call `complete` with:
    - reasoning: Your synthesis of the sub-agent responses and multi-hop reasoning
    - cited_doc_ids: Collect all document IDs mentioned by the sub-agent
    - final_answer: Your final answer in a few words without explanation

    Remember: You orchestrate, the sub-agent retrieves. Stay focused on planning and synthesis.
    """).strip()

    # Create parser
    parser = vf.Parser()

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
