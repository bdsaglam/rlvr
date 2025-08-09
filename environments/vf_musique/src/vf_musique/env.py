import verifiers as vf

from .data import prepare_dataset
from .rewards import (
    citation_reward,
    combined_reward,
    exact_match_reward,
    f1_reward,
    retrieval_recall_reward,
)
from .tools import make_get_tool, make_list_tool, make_retrieve_tool


class MuSiQueEnv(vf.ToolEnv):
    """Custom ToolEnv for MuSiQue with document injection."""

    def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """Set up state and inject documents into tools."""
        state = super().setup_state(state, **kwargs)

        # Get docs from state info
        docs = state.get("info", {}).get("docs", [])
        assert len(docs) > 0, "No docs found in state"

        # Inject documents into tools that need them
        for tool in self.tools:
            if hasattr(tool, "_docs"):
                tool._docs = docs

        return state


# Custom rubric for MuSiQue evaluation
def MuSiQueRubric(parser, **kwargs):
    """Create MuSiQue rubric with all reward functions."""
    reward_funcs = [
        exact_match_reward,
        f1_reward,
        retrieval_recall_reward,
        citation_reward,
        combined_reward,
    ]

    # Combined reward gets weight 1, others are for metrics only
    weights = [0.0, 0.0, 0.0, 0.0, 1.0]

    return vf.Rubric(funcs=reward_funcs, weights=weights, parser=parser, **kwargs)


def load_environment(
    datasets_str: str = "bdsaglam/musique,answerable,train",
    eval_datasets_str: str | None = None,
    retriever_name: str = "hybrid",
    retriever_top_k: int = 3,
    noise_rate: float = 1.0,
    n_jobs: int = 1,
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
        make_retrieve_tool(name=retriever_name, top_k=retriever_top_k),
        make_get_tool(),
        make_list_tool(),
    ]

    # System prompt for MuSiQue
    system_prompt = """Answer the question based on the information provided by tools. You have access to the following tools:
====
{tool_descriptions}
====

For each step:
1. Think through your reasoning inside <think> tags
2. Use tools to retrieve relevant documents
3. Continue until you have found the answer through multi-hop reasoning
4. In the **last** step:
   - Reflect on your previous steps inside <think> tags
   - Cite the documents you used inside <cite> tags by their IDs, e.g. `<cite>1, 2, 3</cite>`
   - Give your final answer inside <answer> tags

- Do not make up tools or arguments that aren't listed.
- Questions require multi-hop reasoning across multiple documents.
- Continue searching until you find all relevant information to answer the question."""

    # Create parser (handles <think>, <cite>, <answer> tags)
    parser = vf.XMLParser(
        fields=["think", "cite", "answer"],
        answer_field="answer",
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
