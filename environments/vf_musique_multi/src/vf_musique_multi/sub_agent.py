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
from pydantic import BaseModel, Field

from .tools import ToolContext, make_retrieve_tool

set_tracing_disabled(disabled=True)


class CustomModelProvider(ModelProvider):
    def __init__(self, client: AsyncOpenAI | None = None):
        if client is None:
            client = AsyncOpenAI()
        self.client = client

    def get_model(self, model_name: str | None) -> Model:
        return OpenAIChatCompletionsModel(
            model=model_name or os.getenv("SUB_AGENT_OPENAI_MODEL_NAME"), openai_client=self.client
        )


class Citation(BaseModel):
    doc_id: str = Field(description="The ID of the document you cited")
    doc_title: str = Field(description="The title of the document you cited")
    quote: str = Field(description="The quote from the document you cited")


class QuestionAnsweringResult(BaseModel):
    reasoning: str = Field(description="Your reasoning based on retrieved documents")
    final_answer: str = Field(description="Your final answer in a few words")
    citations: list[Citation] = Field(
        default_factory=list, description="List of citations for your reasoning and answer"
    )


def make_sub_agent_tool(retriever: str = "hybrid", model: str = None, top_n: int = 1) -> Callable:
    """Create a tool that delegates to the sub-agent for document retrieval and answering."""
    retrieve_documents = make_retrieve_tool(name=retriever, top_n=top_n)

    # Initialize the sub-agent
    sub_agent = Agent(
        name="SubQuestionAgent",
        instructions="""
            You are a retrieval augmented question answering specialist. Your job is to:
            1. Use the retrieve_documents tool to find relevant information
            2. Answer the specific question based ONLY on the retrieved documents
            3. Provide reasoning based on the documents you found
            4. Cite the document IDs you used in your reasoning
        
            You must keep searching until you find the answer.
            Your final response should include the following fields:
            **Reasoning:** [Your reasoning based on retrieved documents]
            **Final Answer:** [Final answer to the question]
            **Citations:** [List of citations for your reasoning and answer, include the document ID and title of the document you cited]
            """,
        tools=[function_tool(retrieve_documents)],
        # output_type=QuestionAnsweringResult,
        model=model,
    )
    sub_agent_base_url = (
        os.getenv("SUB_AGENT_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    )
    sub_agent_api_key = os.getenv("SUB_AGENT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    client = AsyncOpenAI(base_url=sub_agent_base_url, api_key=sub_agent_api_key)
    model_provider = CustomModelProvider(client=client)
    run_config = RunConfig(model_provider=model_provider)

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
        try:
            result = await Runner.run(
                sub_agent,
                input=sub_question,
                max_turns=5,
                context=wrapper.context,
                run_config=run_config,
            )
            return result.final_output
        except Exception as e:
            return f"Error running sub-agent: {str(e)}"

    return answer_subquestion
