"""Sub-agent for handling document retrieval and question answering."""

from agents import Agent, RunContextWrapper, Runner

from .llm import get_default_model
from .tools import ToolContext, make_retrieve_tool


class RagQaAgent:
    """Sub-agent that handles document retrieval and focused question answering."""

    def __init__(self, retriever: str = "hybrid", model: str = None):
        self.retriever = retriever
        self.model = model or get_default_model()

        # Reuse the existing retrieve tool
        retrieve_documents = make_retrieve_tool(name=retriever, default_top_n=1)

        # Create the sub-agent
        self.agent = Agent(
            name="RagQaAgent",
            instructions="""
            You are a retrieval augmented question answering specialist. Your job is to:
            1. Use the retrieve_documents tool to find relevant information
            2. Answer the specific sub-question based ONLY on the retrieved documents
            3. Provide reasoning based on the documents you found
            4. Cite the document IDs you used in your reasoning

            Format your response as:
            **Reasoning:** [Your reasoning based on retrieved documents]
            **Answer:** [Direct answer to the sub-question]
            **Cited Documents:** [List of document IDs used]

            If you cannot find sufficient information, say so clearly.
            """,
            tools=[retrieve_documents],
            model=self.model,
        )

    async def answer_subquestion(self, ctx: ToolContext, sub_question: str) -> str:
        """Answer a sub-question using document retrieval with proper context."""
        try:
            # Run the agent with the context
            result = await Runner.run(
                self.agent,
                input=f"Answer this sub-question: {sub_question}",
                max_turns=5,
                context=ctx,
            )
            return result.final_output
        except Exception as e:
            return f"Error answering sub-question: {str(e)}"
