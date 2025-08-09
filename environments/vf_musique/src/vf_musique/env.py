import json
from typing import Tuple

import verifiers as vf
from verifiers.types import ChatCompletionMessageToolCall, Message, Messages, State

from .data import prepare_dataset
from .rewards import (
    citation_reward,
    combined_reward,
    exact_match_reward,
    f1_reward,
    retrieval_recall_reward,
)
from .tools import make_get_tool, make_retrieve_tool


class MuSiQueEnv(vf.ToolEnv):
    """Custom ToolEnv for MuSiQue with document injection via tool_args."""

    def env_response(self, messages: Messages, state: State, **kwargs) -> Tuple[Messages, State]:
        assert isinstance(messages, list)
        assert "tool_calls" in messages[-1]
        tool_messages = []
        for tool_call in messages[-1]["tool_calls"]:
            assert isinstance(tool_call, ChatCompletionMessageToolCall)
            tool_name: str = tool_call.function.name
            tool_args: str = tool_call.function.arguments
            tool_args = self._inject_docs(tool_args, state)
            tool_call_id: str = tool_call.id or ""
            tool_message: Message = self.call_tool(tool_name, tool_args, tool_call_id)
            tool_messages.append(tool_message)
        return tool_messages, state

    def _inject_docs(self, tool_args: str, state: State) -> str:
        """Inject docs into tool_args."""
        docs = state["info"]["docs"]
        payload = json.loads(tool_args)
        payload["docs"] = docs
        return json.dumps(payload)


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
        make_retrieve_tool(name=retriever_name),
        make_get_tool(),
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
An example for your final message:
```
<think>
[your thinking here]
</think> 
<cite>
[IDs of the documents that back your answer, e.g. 5, 3, 2]
</cite>
<answer>
[your final answer here without any additional text]
</answer>
```

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
