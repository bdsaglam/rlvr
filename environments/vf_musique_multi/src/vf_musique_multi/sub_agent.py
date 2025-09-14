"""Sub-agent for handling document retrieval and question answering."""

import os
from typing import Callable

from agents import (
    Agent,
    Model,
    ModelProvider,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from openai import AsyncOpenAI

from .tools import ToolContext, make_retrieve_tool


set_tracing_disabled(disabled=True)


class CustomModelProvider(ModelProvider):
    def __init__(self, client: AsyncOpenAI | None = None):
        if client is None:
            client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"), api_key=os.getenv("OPENAI_API_KEY"))
        self.client = client

    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(model=model_name or os.getenv("OPENAI_MODEL_NAME"), openai_client=self.client)


def make_sub_agent_tool(retriever: str = "hybrid", model: str = None, default_top_n=1) -> Callable:
    """Create a tool that delegates to the sub-agent for document retrieval and answering."""
    retrieve_documents = make_retrieve_tool(name=retriever, default_top_n=default_top_n)

    # Initialize the sub-agent
    sub_agent = Agent(
        name="SubQuestionAgent",
        instructions="""
            You are a retrieval augmented question answering specialist. Your job is to:
            1. Use the retrieve_documents tool to find relevant information
            2. Answer the specific sub-question based ONLY on the retrieved documents
            3. Provide reasoning based on the documents you found
            4. Cite the document IDs you used in your reasoning

            Format your response as:
            **Reasoning:** [Your reasoning based on retrieved documents]
            **Answer:** [Direct answer to the sub-question]
            **Cited Documents:** [List of document IDs your reasoning and answer are based on]

            If you cannot find sufficient information, say so clearly.
            """,
        tools=[function_tool(retrieve_documents)],
        model=model,
    )
    model_provider = CustomModelProvider()

    async def answer_subquestion(wrapper: RunContextWrapper[ToolContext], sub_question: str) -> str:
        """
        Delegate a sub-question to the retrieval sub-agent.

        The sub-agent will:
        1. Use document retrieval to find relevant information
        2. Answer the sub-question based on retrieved documents
        3. Provide reasoning and document citations

        Args:
            sub_question: A focused sub-question to be answered using document retrieval.

        Returns:
            The sub-agent's response with reasoning, answer, and cited documents.
        """
        # Simply await the async sub-agent method with context
        try:
            result = await Runner.run(
                sub_agent,
                input=f"Answer this sub-question: {sub_question}",
                max_turns=5,
                context=wrapper.context,
                run_config=RunConfig(model_provider=model_provider),
            )
            return result.final_output
        except Exception as e:
            return f"Error running sub-agent: {str(e)}"

    return answer_subquestion
